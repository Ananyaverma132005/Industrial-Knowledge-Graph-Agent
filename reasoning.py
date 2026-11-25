"""Graph reasoning over failures: root cause tracing, troubleshooting paths, impact analysis
and spare-part availability.

Each function walks the knowledge graph (FailureMode -[:CAUSED_BY]-> FailureMode,
Ticket -[:INDICATES]-> FailureMode, FailureMode -[:RESOLVED_BY]-> SOP,
FailureMode -[:NEEDS_SPARE]-> SparePart, Machine -[:FEEDS]-> Machine) and returns the answer
text, the equivalent Cypher query and the node ids to highlight on the dashboard.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def load_phrases() -> dict:
    """Symptom phrases per failure mode, used to spot the symptom in a question."""
    try:
        kb = json.loads(config.FAILURE_KB.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {f["id"]: [p.lower() for p in f.get("phrases", [])] for f in kb.get("failure_modes", [])}

INTENT_PATTERNS = [
    ("impact", r"\bimpact|affect(ed|s)?\b|downstream|knock[- ]on|what happens if|consequence|if .+ (fails|stops|goes down|breaks)"),
    ("root_cause", r"root cause|what causes|why (does|do|did|is|are)|cause[sd]? of|reason for|caused by"),
    ("troubleshoot", r"troubleshoot|diagnos|fault[- ]?find|step[- ]by[- ]step check"),
    ("spares", r"\bspares?\b|in stock|out of stock|stock|inventory|lead time|replacement part"),
]

CYPHER = {
    "root_cause": (
        "MATCH (s:FailureMode) WHERE s.id IN $symptoms\n"
        "MATCH path = (s)-[:CAUSED_BY*1..2]->(c:FailureMode)\n"
        "OPTIONAL MATCH (t:Ticket)-[:INDICATES]->(c)\n"
        "RETURN s.name AS symptom, c.name AS cause, length(path) AS depth,\n"
        "       count(DISTINCT t) AS evidence, collect(DISTINCT t.id) AS tickets\n"
        "ORDER BY evidence DESC, depth"
    ),
    "troubleshoot": (
        "MATCH (s:FailureMode) WHERE s.id IN $symptoms\n"
        "OPTIONAL MATCH (s)-[:CAUSED_BY*1..2]->(c:FailureMode)<-[:INDICATES]-(t:Ticket)\n"
        "WITH s, c, count(t) AS evidence ORDER BY evidence DESC\n"
        "OPTIONAL MATCH (s)-[:RESOLVED_BY]->(sop:SOP)\n"
        "OPTIONAL MATCH (s)-[:NEEDS_SPARE]->(sp:SparePart)\n"
        "RETURN s.name, collect(DISTINCT c.name) AS checks_in_order,\n"
        "       collect(DISTINCT sop.title) AS procedures,\n"
        "       collect(DISTINCT {spare: sp.name, stock: sp.stock, lead_days: sp.lead_time_days}) AS spares"
    ),
    "impact": (
        "MATCH (m:Machine) WHERE m.id IN $mids\n"
        "OPTIONAL MATCH (m)-[:FEEDS*1..]->(down:Machine)\n"
        "OPTIONAL MATCH (m)-[:HAS_PART]->(p:Part)-[:USES_SPARE]->(sp:SparePart)\n"
        "OPTIONAL MATCH (p)<-[:REPORTS_FAILURE_ON]-(t:Ticket)\n"
        "OPTIONAL MATCH (tech:Technician)-[:MAINTAINED_BY]->(m)\n"
        "RETURN m.name AS machine, collect(DISTINCT down.name) AS downstream,\n"
        "       collect(DISTINCT {spare: sp.name, stock: sp.stock, lead_days: sp.lead_time_days}) AS spares,\n"
        "       count(DISTINCT t) AS past_failures, collect(DISTINCT tech.name) AS technicians"
    ),
    "spares": (
        "MATCH (m:Machine)-[:HAS_PART]->(p:Part)-[:USES_SPARE]->(sp:SparePart)\n"
        "WHERE (size($mids) = 0 OR m.id IN $mids) AND (size($pnames) = 0 OR toLower(p.name) IN $pnames)\n"
        "RETURN DISTINCT sp.id, sp.name, sp.stock, sp.lead_time_days, sp.location ORDER BY sp.stock"
    ),
}


def detect_intent(query: str) -> str | None:
    q = query.lower()
    for name, pattern in INTENT_PATTERNS:
        if re.search(pattern, q):
            return name
    return None


class _Graph:
    """Adjacency view of the store, so the same reasoning runs on Neo4j data or the in-memory graph."""

    def __init__(self, store):
        self.node = {}
        for label in ("Machine", "Part", "Ticket", "SOP", "Technician", "FailureMode", "SparePart"):
            for n in store.nodes(label):
                self.node[n["id"]] = {**n, "_label": label}
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        self.edge_props = {}
        for rel in ("HAS_PART", "REPORTS_FAILURE_ON", "MAINTAINED_BY", "FEEDS", "CAUSED_BY", "INDICATES",
                    "RESOLVED_BY", "NEEDS_SPARE", "USES_SPARE", "AFFECTS"):
            for e in store.edges(rel):
                self.out[(e["source"], rel)].append(e["target"])
                self.inc[(e["target"], rel)].append(e["source"])
                self.edge_props[(e["source"], rel, e["target"])] = e.get("props") or {}

    def name(self, nid: str) -> str:
        n = self.node.get(nid, {})
        return n.get("name") or n.get("title") or nid

    def label(self, nid: str) -> str:
        return self.node.get(nid, {}).get("_label", "")

    def ids(self, label: str) -> list[str]:
        return [i for i, n in self.node.items() if n["_label"] == label]


# ---------------------------------------------------------------------------- symptom lookup
def _find_symptoms(g: _Graph, query: str, mids: list[str], pnames: list[str], kb_phrases: dict) -> list[str]:
    found = _match_symptoms(g, query.lower(), mids, pnames, kb_phrases)
    # If one match is itself a cause of another (misalignment -> breakage), keep the end symptom only.
    return [f for f in found if not any(f in g.out[(o, "CAUSED_BY")] for o in found if o != f)]


def _match_symptoms(g: _Graph, q: str, mids: list[str], pnames: list[str], kb_phrases: dict) -> list[str]:
    found = [fid for fid in g.ids("FailureMode")
             if g.name(fid).lower() in q or any(p in q for p in kb_phrases.get(fid, []))]
    # Prefer the most specific match: drop modes whose name is contained in another match.
    found = [f for f in found if not any(f != o and g.name(f).lower() in g.name(o).lower() for o in found)]
    if found:
        return found
    if pnames:  # a part was named: failure modes that affect it and have causes to trace
        hits = set()
        for fid in g.ids("FailureMode"):
            parts = {g.name(p).lower() for p in g.out[(fid, "AFFECTS")]}
            if parts & set(pnames) and g.out[(fid, "CAUSED_BY")]:
                hits.add(fid)
        return sorted(hits)
    if mids:  # a machine was named: symptoms seen in its tickets
        hits = set()
        for tid in g.ids("Ticket"):
            if g.node[tid].get("machine_id") in mids:
                hits.update(f for f in g.out[(tid, "INDICATES")] if g.out[(f, "CAUSED_BY")])
        return sorted(hits)
    return []


def _tickets_for(g: _Graph, fid: str, mids: list[str]) -> list[str]:
    tickets = g.inc[(fid, "INDICATES")]
    if mids:
        tickets = [t for t in tickets if g.node[t].get("machine_id") in mids]
    return sorted(tickets)


def _ranked_causes(g: _Graph, symptom: str, mids: list[str]) -> list[dict]:
    if not mids:  # count evidence only on machines that have a part this symptom affects
        affected = set(g.out[(symptom, "AFFECTS")])
        mids = [m for m in g.ids("Machine") if affected & set(g.out[(m, "HAS_PART")])]
    rows, seen = [], set()
    frontier = [(c, 1, None) for c in g.out[(symptom, "CAUSED_BY")]]
    while frontier:
        cause, depth, via = frontier.pop(0)
        if cause in seen or depth > 2:
            continue
        seen.add(cause)
        tickets = _tickets_for(g, cause, mids)
        rows.append({"id": cause, "cause": g.name(cause), "depth": depth, "via": via,
                     "evidence": len(tickets), "tickets": tickets})
        frontier += [(c, depth + 1, g.name(cause)) for c in g.out[(cause, "CAUSED_BY")]]
    rows.sort(key=lambda r: (-r["evidence"], r["depth"], r["cause"]))
    return rows


def _spare_line(g: _Graph, sp: str) -> str:
    n = g.node[sp]
    stock = n.get("stock", 0)
    status = f"{stock} in stock ({n.get('location')})" if stock else f"OUT OF STOCK, lead time {n.get('lead_time_days')} days"
    return f"**{n['name']}**: {status}"


# ---------------------------------------------------------------------------- reasoning
def root_cause(g: _Graph, symptoms: list[str], mids: list[str]) -> tuple[str, list[dict], list[str]]:
    lines, rows, hl = [], [], set(symptoms)
    for s in symptoms:
        causes = _ranked_causes(g, s, mids)
        own = _tickets_for(g, s, mids)
        lines.append(f"**Root causes of {g.name(s)}** ({len(own)} ticket{'s report' if len(own) != 1 else ' reports'} this symptom), "
                     "ranked by how often each cause appears in past tickets:")
        for i, c in enumerate(causes, 1):
            where = "direct cause" if c["depth"] == 1 else f"underlying cause, via {c['via']}"
            ev = f"{c['evidence']} ticket{'s' if c['evidence'] != 1 else ''}"
            tk = f" ({', '.join(c['tickets'])})" if c["tickets"] else ""
            lines.append(f"{i}. **{c['cause']}** ({where}): {ev}{tk}")
            hl.add(c["id"])
            hl.update(c["tickets"])
        rows += [{"symptom": g.name(s), **c} for c in causes]
    return "\n".join(lines), rows, sorted(hl)


def troubleshoot(g: _Graph, symptoms: list[str], mids: list[str]) -> tuple[str, list[dict], list[str]]:
    lines, rows, hl = [], [], set(symptoms)
    for s in symptoms:
        causes = _ranked_causes(g, s, mids)
        sops = g.out[(s, "RESOLVED_BY")]
        spares = g.out[(s, "NEEDS_SPARE")]
        lines.append(f"**Troubleshooting path for {g.name(s)}**")
        step = 1
        lines.append(f"{step}. Isolate the machine first: follow **SOP-SAF-01 Lockout/Tagout**.")
        hl.add("SOP-SAF-01")
        for c in causes[:3]:
            step += 1
            ev = f"{c['evidence']} past ticket{'s' if c['evidence'] != 1 else ''}"
            lines.append(f"{step}. Check for **{c['cause']}** ({ev}).")
            hl.add(c["id"])
        for sop in sops:
            step += 1
            lines.append(f"{step}. Apply **{g.name(sop)}**.")
            hl.add(sop)
        if spares:
            step += 1
            lines.append(f"{step}. Spare parts that may be needed: " + "; ".join(_spare_line(g, sp) for sp in spares) + ".")
            hl.update(spares)
        rows.append({"symptom": g.name(s), "checks": [c["cause"] for c in causes[:3]],
                     "procedures": [g.name(x) for x in sops], "spares": [g.name(x) for x in spares]})
    return "\n".join(lines), rows, sorted(hl)


def impact(g: _Graph, mids: list[str], pnames: list[str]) -> tuple[str, list[dict], list[str]]:
    if not mids and pnames:  # a part was named: look at the machines that have it
        mids = sorted({m for m in g.ids("Machine") for p in g.out[(m, "HAS_PART")] if g.name(p).lower() in pnames})
    lines, rows, hl = [], [], set(mids)
    order = _line_order(g)
    for m in mids:
        down = _reachable(g, m)
        up = [x for x in order if m in _reachable(g, x)]
        parts = g.out[(m, "HAS_PART")]
        if pnames:
            parts = [p for p in parts if g.name(p).lower() in pnames] or parts
        failures = sorted({t for p in parts for t in g.inc[(p, "REPORTS_FAILURE_ON")]})
        spares = sorted({sp for p in parts for sp in g.out[(p, "USES_SPARE")]})
        techs = g.inc[(m, "MAINTAINED_BY")]
        lines.append(f"**Impact if {g.name(m)} ({m}) stops**")
        if down:
            lines.append(f"- Downstream machines that run out of work: {', '.join(f'{g.name(d)} ({d})' for d in down)}.")
        else:
            lines.append("- It is the last machine on the line, so no other machine is starved, but finished output stops.")
        if up:
            lines.append(f"- Upstream machines that must slow down or stop: {', '.join(f'{g.name(u)} ({u})' for u in up)}.")
        share = round(100 * (len(down) + 1) / max(1, len(order)))
        lines.append(f"- {len(down) + 1} of {len(order)} machines on Assembly Line 1 lose output ({share}%).")
        lines.append(f"- Failure history: {len(failures)} past tickets on its parts.")
        if spares:
            missing = [sp for sp in spares if not g.node[sp].get("stock")]
            lines.append("- Spare parts: " + "; ".join(_spare_line(g, sp) for sp in spares) + ".")
            if missing:
                worst = max(g.node[sp].get("lead_time_days", 0) for sp in missing)
                lines.append(f"- Risk: {len(missing)} spare part(s) out of stock, so a repair could wait up to {worst} days.")
        if techs:
            lines.append(f"- Contact: {', '.join(g.name(t) for t in techs)}.")
        hl.update(down + spares + techs)
        rows.append({"machine": g.name(m), "downstream": [g.name(d) for d in down], "upstream": [g.name(u) for u in up],
                     "past_failures": len(failures), "spares": [g.name(s) for s in spares]})
    return "\n".join(lines), rows, sorted(hl)


def spares(g: _Graph, mids: list[str], pnames: list[str], query: str) -> tuple[str, list[dict], list[str]]:
    q = query.lower()
    candidates = g.ids("SparePart")
    if mids or pnames:
        parts = [p for m in (mids or g.ids("Machine")) for p in g.out[(m, "HAS_PART")]
                 if not pnames or g.name(p).lower() in pnames]
        candidates = sorted({sp for p in parts for sp in g.out[(p, "USES_SPARE")]})
    if re.search(r"out of stock|low stock|shortage|missing", q):
        candidates = [sp for sp in candidates if g.node[sp].get("stock", 0) <= 1]
    candidates.sort(key=lambda sp: (g.node[sp].get("stock", 0), g.name(sp)))
    if not candidates:
        return "No matching spare parts found.", [], []
    scope = ", ".join(g.name(m) for m in mids) or ", ".join(p.title() for p in pnames)
    title = "**Spare parts" + (f" for {scope}" if scope else "") + "**"
    title += " (low or out of stock)" if re.search(r"out of stock|low stock|shortage|missing", q) else ""
    lines = [title + ":"] + [f"- {_spare_line(g, sp)}" for sp in candidates]
    rows = [{"id": sp, "name": g.name(sp), "stock": g.node[sp].get("stock"), "lead_time_days": g.node[sp].get("lead_time_days")} for sp in candidates]
    return "\n".join(lines), rows, candidates


def _reachable(g: _Graph, start: str) -> list[str]:
    seen, stack = [], list(g.out[(start, "FEEDS")])
    while stack:
        n = stack.pop(0)
        if n not in seen:
            seen.append(n)
            stack += g.out[(n, "FEEDS")]
    return seen


def _line_order(g: _Graph) -> list[str]:
    machines = g.ids("Machine")
    starts = [m for m in machines if not g.inc[(m, "FEEDS")]]
    order = []
    for s in starts:
        order += [s] + [x for x in _reachable(g, s) if x not in order]
    return order + [m for m in machines if m not in order]


def answer(query: str, store, mids: list[str], pnames: list[str], kb_phrases: dict | None = None) -> dict | None:
    """Return an answer dict for reasoning questions, or None if the question is not one."""
    intent = detect_intent(query)
    if not intent:
        return None
    g = _Graph(store)
    params = {"mids": mids, "pnames": pnames}
    if intent in ("root_cause", "troubleshoot"):
        symptoms = _find_symptoms(g, query, mids, pnames, kb_phrases if kb_phrases is not None else load_phrases())
        if not symptoms:
            names = sorted(g.name(f) for f in g.ids("FailureMode") if g.out[(f, "CAUSED_BY")])
            return {"answer": "Please name a symptom, for example: " + ", ".join(names[:6]) + ".",
                    "cypher": CYPHER[intent], "params": params, "rows": [], "intent": intent, "highlight": []}
        params["symptoms"] = symptoms
        text, rows, hl = (root_cause if intent == "root_cause" else troubleshoot)(g, symptoms, mids)
    elif intent == "impact":
        if not mids and not pnames:
            return {"answer": "Please name a machine or part, for example: What is the impact if the belt conveyor fails?",
                    "cypher": CYPHER[intent], "params": params, "rows": [], "intent": intent, "highlight": []}
        text, rows, hl = impact(g, mids, pnames)
    else:
        text, rows, hl = spares(g, mids, pnames, query)
    return {"answer": text, "cypher": CYPHER[intent], "params": params, "rows": rows, "intent": intent, "highlight": hl}
