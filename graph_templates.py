"""Deterministic graph question answering.

Used when Mistral is unavailable, when the LLM-generated Cypher fails, or when it returns
nothing. Each intent has a parameterised Cypher query (run on Neo4j) and an equivalent
Python evaluator (run on the in-memory store), so the UI can always show the Cypher behind
an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------
GENERIC_TOKENS = {"machine", "cnc", "robot", "arm", "the", "and", "system"}
PRONOUNS = re.compile(r"\b(it|its|this|that|they|them|those|these|same)\b")


def _norm(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") else tok


def _tokens(text: str) -> set[str]:
    return {_norm(t) for t in re.findall(r"[a-z0-9]+", text.lower())}


@dataclass
class Entities:
    machines: list[str] = field(default_factory=list)   # machine ids
    parts: list[str] = field(default_factory=list)      # lower-case part names
    carried_over: bool = False


def resolve_entities(query: str, machines: list[dict], part_names: list[str], last: Entities | None) -> Entities:
    q = query.lower()
    parts = []
    for name in sorted(set(part_names), key=len, reverse=True):  # longest first: "servo motor" before "motor"
        if re.search(rf"\b{re.escape(name.lower())}s?\b", q):
            parts.append(name.lower())
            q = re.sub(rf"\b{re.escape(name.lower())}s?\b", " ", q)  # don't let "hydraulic seal" match the press

    q_tokens = _tokens(q)
    scores = {}
    for m in machines:
        score = 0
        if re.search(rf"\b{m['id'].lower()}\b", q) or m["name"].lower() in q:
            score += 3
        distinctive = _tokens(m["name"]) - GENERIC_TOKENS
        score += 2 * len(distinctive & q_tokens)
        if _norm(m["type"].lower()) in q_tokens:
            score += 1
        if score:
            scores[m["id"]] = score
    best = max(scores.values(), default=0)
    mids = sorted(mid for mid, s in scores.items() if s == best) if best else []

    ents = Entities(machines=mids, parts=parts)
    if not mids and not parts and last and PRONOUNS.search(query.lower()):
        ents = Entities(machines=list(last.machines), parts=list(last.parts), carried_over=True)
    return ents


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
COUNT_LABELS = [
    ("Ticket", r"ticket|failure|incident|breakdown|fault"),
    ("Part", r"\bparts?\b|component"),
    ("Sensor", r"sensor"),
    ("Technician", r"technician|engineer|people|staff"),
    ("SOP", r"\bsops?\b|procedure|manual"),
    ("Machine", r"machine|asset|equipment"),
]


def detect_intent(query: str, ents: Entities) -> str:
    q = query.lower()
    if re.search(r"at[- ]risk|most (fail|ticket|problem|issue)|frequent|predict|high(est)? (failure|risk)|failure[- ]prone|critical part|repeat", q):
        return "at_risk"
    if re.search(r"how many|count|number of|\btotal\b", q):
        return "count"
    if re.search(r"\bwho\b|technician|maintain|maintenance date|last (serviced|maintenance)|assigned|responsible", q):
        return "technician"
    if re.search(r"sensor|threshold|reading|temperature|vibration|pressure|telemetry", q):
        return "sensors"
    if re.search(r"ticket|failure|\bfail|issue|problem|fault|incident|breakdown|history|reported|complaint", q):
        return "tickets"
    if re.search(r"\bparts?\b|component|consist|made of|contain|which machines? (has|have|use)", q) or (ents.parts and not ents.machines):
        return "parts"
    if re.search(r"status|state|condition|operational|degraded|under maintenance|running", q):
        return "status"
    if ents.machines:
        return "overview"
    return "machines"


# ---------------------------------------------------------------------------
# Cypher per intent (run on Neo4j)
# ---------------------------------------------------------------------------
M_FILTER = "(size($mids) = 0 OR m.id IN $mids)"
P_FILTER = "(size($pnames) = 0 OR toLower(p.name) IN $pnames)"

CYPHER = {
    "machines": "MATCH (m:Machine) RETURN m.id AS id, m.name AS name, m.type AS type, m.status AS status ORDER BY m.id",
    "status": f"MATCH (m:Machine) WHERE {M_FILTER} RETURN m.id AS id, m.name AS name, m.status AS status ORDER BY m.id",
    "overview": (
        "MATCH (m:Machine) WHERE m.id IN $mids\n"
        "OPTIONAL MATCH (m)-[:HAS_PART]->(p:Part)\n"
        "WITH m, collect(DISTINCT p.name) AS parts\n"
        "OPTIONAL MATCH (m)-[:HAS_SENSOR]->(s:Sensor)\n"
        "WITH m, parts, collect(DISTINCT s.type) AS sensors\n"
        "OPTIONAL MATCH (t:Technician)-[:MAINTAINED_BY]->(m)\n"
        "RETURN m.id AS id, m.name AS name, m.type AS type, m.status AS status, parts, sensors,\n"
        "       collect(DISTINCT t.name) AS technicians ORDER BY m.id"
    ),
    "parts": (
        f"MATCH (m:Machine)-[:HAS_PART]->(p:Part) WHERE {M_FILTER} AND {P_FILTER}\n"
        "RETURN m.id AS machine_id, m.name AS machine, p.id AS id, p.name AS part ORDER BY m.id, p.name"
    ),
    "sensors": (
        f"MATCH (m:Machine)-[:HAS_SENSOR]->(s:Sensor) WHERE {M_FILTER}\n"
        "RETURN m.id AS machine_id, m.name AS machine, s.id AS id, s.type AS type,\n"
        "       s.reading AS reading, s.threshold AS threshold, s.unit AS unit ORDER BY m.id, s.type"
    ),
    "tickets": (
        f"MATCH (m:Machine)-[:HAS_PART]->(p:Part)<-[:REPORTS_FAILURE_ON]-(t:Ticket)\nWHERE {M_FILTER} AND {P_FILTER}\n"
        "RETURN m.id AS machine_id, m.name AS machine, p.id AS part_id, p.name AS part,\n"
        "       t.id AS id, t.text AS text, t.timestamp AS timestamp ORDER BY t.timestamp DESC"
    ),
    "technician": (
        f"MATCH (t:Technician)-[r:MAINTAINED_BY]->(m:Machine) WHERE {M_FILTER}\n"
        "RETURN t.id AS id, t.name AS technician, t.specialization AS specialization,\n"
        "       m.id AS machine_id, m.name AS machine, r.last_maintenance AS last_maintenance ORDER BY m.id"
    ),
    "at_risk": (
        "MATCH (m:Machine)-[:HAS_PART]->(p:Part)<-[:REPORTS_FAILURE_ON]-(t:Ticket)\n"
        f"WHERE {M_FILTER}\n"
        "WITH m, p, count(t) AS ticket_count, collect(t.id) AS tickets\n"
        "WHERE ticket_count >= 2\n"
        "RETURN p.id AS part_id, p.name AS part, m.id AS machine_id, m.name AS machine, ticket_count, tickets\n"
        "ORDER BY ticket_count DESC, part_id"
    ),
}

COUNT_CYPHER = {
    "Machine": "MATCH (n:Machine) RETURN count(n) AS count",
    "Technician": "MATCH (n:Technician) RETURN count(n) AS count",
    "SOP": "MATCH (n:SOP) OPTIONAL MATCH (n)-[:APPLIES_TO]->(m:Machine) WITH n, m WHERE " + M_FILTER + " RETURN count(DISTINCT n) AS count",
    "Part": f"MATCH (m:Machine)-[:HAS_PART]->(p:Part) WHERE {M_FILTER} AND {P_FILTER} RETURN count(p) AS count",
    "Sensor": f"MATCH (m:Machine)-[:HAS_SENSOR]->(n:Sensor) WHERE {M_FILTER} RETURN count(n) AS count",
    "Ticket": (f"MATCH (m:Machine)-[:HAS_PART]->(p:Part)<-[:REPORTS_FAILURE_ON]-(n:Ticket) "
               f"WHERE {M_FILTER} AND {P_FILTER} RETURN count(DISTINCT n) AS count"),
}


# ---------------------------------------------------------------------------
# Python evaluators (run on the in-memory store, same row shapes as the Cypher)
# ---------------------------------------------------------------------------
class _Index:
    def __init__(self, store):
        self.machines = {m["id"]: m for m in store.nodes("Machine")}
        self.parts = {p["id"]: p for p in store.nodes("Part")}
        self.sensors = {s["id"]: s for s in store.nodes("Sensor")}
        self.tickets = {t["id"]: t for t in store.nodes("Ticket")}
        self.techs = {t["id"]: t for t in store.nodes("Technician")}
        self.sops = {s["id"]: s for s in store.nodes("SOP")}
        self.e = {r: store.edges(r) for r in ("HAS_PART", "HAS_SENSOR", "REPORTS_FAILURE_ON", "MAINTAINED_BY", "APPLIES_TO")}


def _python_rows(intent: str, store, mids: list[str], pnames: list[str], label: str | None = None) -> list[dict]:
    ix = _Index(store)
    m_ok = (lambda mid: not mids or mid in mids)
    p_ok = (lambda pid: not pnames or ix.parts[pid]["name"].lower() in pnames)
    has_part = [(e["source"], e["target"]) for e in ix.e["HAS_PART"]]
    failure = [(e["source"], e["target"]) for e in ix.e["REPORTS_FAILURE_ON"]]

    if intent == "machines":
        return [{"id": m["id"], "name": m["name"], "type": m["type"], "status": m["status"]} for m in sorted(ix.machines.values(), key=lambda m: m["id"])]
    if intent == "status":
        return [{"id": m["id"], "name": m["name"], "status": m["status"]} for m in sorted(ix.machines.values(), key=lambda m: m["id"]) if m_ok(m["id"])]
    if intent == "overview":
        rows = []
        for mid in mids:
            m = ix.machines[mid]
            rows.append({
                "id": mid, "name": m["name"], "type": m["type"], "status": m["status"],
                "parts": [ix.parts[p]["name"] for s, p in has_part if s == mid],
                "sensors": [ix.sensors[e["target"]]["type"] for e in ix.e["HAS_SENSOR"] if e["source"] == mid],
                "technicians": [ix.techs[e["source"]]["name"] for e in ix.e["MAINTAINED_BY"] if e["target"] == mid],
            })
        return rows
    if intent == "parts":
        rows = [{"machine_id": s, "machine": ix.machines[s]["name"], "id": p, "part": ix.parts[p]["name"]}
                for s, p in has_part if m_ok(s) and p_ok(p)]
        return sorted(rows, key=lambda r: (r["machine_id"], r["part"]))
    if intent == "sensors":
        rows = []
        for e in ix.e["HAS_SENSOR"]:
            if m_ok(e["source"]):
                s = ix.sensors[e["target"]]
                rows.append({"machine_id": e["source"], "machine": ix.machines[e["source"]]["name"], "id": s["id"],
                             "type": s["type"], "reading": s["reading"], "threshold": s["threshold"], "unit": s["unit"]})
        return sorted(rows, key=lambda r: (r["machine_id"], r["type"]))
    if intent == "tickets":
        machine_of = {p: s for s, p in has_part}
        rows = []
        for t, p in failure:
            mid = machine_of[p]
            if m_ok(mid) and p_ok(p):
                tk = ix.tickets[t]
                rows.append({"machine_id": mid, "machine": ix.machines[mid]["name"], "part_id": p,
                             "part": ix.parts[p]["name"], "id": t, "text": tk["text"], "timestamp": tk["timestamp"]})
        return sorted(rows, key=lambda r: r["timestamp"], reverse=True)
    if intent == "technician":
        rows = []
        for e in ix.e["MAINTAINED_BY"]:
            if m_ok(e["target"]):
                t = ix.techs[e["source"]]
                rows.append({"id": t["id"], "technician": t["name"], "specialization": t["specialization"],
                             "machine_id": e["target"], "machine": ix.machines[e["target"]]["name"],
                             "last_maintenance": e["props"].get("last_maintenance")})
        return sorted(rows, key=lambda r: r["machine_id"])
    if intent == "at_risk":
        return [{k: r[k] for k in ("part_id", "part", "machine_id", "machine", "ticket_count", "tickets")}
                for r in store.at_risk_parts() if m_ok(r["machine_id"])]
    if intent == "count":
        if label == "Machine":
            n = len(ix.machines)
        elif label == "Technician":
            n = len(ix.techs)
        elif label == "SOP":
            n = len({e["source"] for e in ix.e["APPLIES_TO"] if m_ok(e["target"])}) if mids else len(ix.sops)
        elif label == "Part":
            n = sum(1 for s, p in has_part if m_ok(s) and p_ok(p))
        elif label == "Sensor":
            n = sum(1 for e in ix.e["HAS_SENSOR"] if m_ok(e["source"]))
        else:  # Ticket
            machine_of = {p: s for s, p in has_part}
            n = len({t for t, p in failure if m_ok(machine_of[p]) and p_ok(p)})
        return [{"count": n}]
    raise ValueError(intent)


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------
PLURAL = {"Machine": "machines", "Part": "parts", "Sensor": "sensors", "Ticket": "maintenance tickets",
          "Technician": "technicians", "SOP": "SOP sections"}


def _scope(mids, pnames, machines_by_id) -> str:
    bits = []
    if pnames:
        bits.append("for " + ", ".join(p.title() for p in pnames))
    if mids:
        bits.append("on " + ", ".join(f"{machines_by_id[m]['name']} ({m})" for m in mids))
    return (" " + " ".join(bits)) if bits else ""


def format_answer(intent: str, rows: list[dict], mids, pnames, machines_by_id, label=None) -> str:
    scope = _scope(mids, pnames, machines_by_id)
    if intent == "count":
        return f"There are **{rows[0]['count']} {PLURAL[label]}**{scope} in the knowledge graph."
    if not rows:
        return f"I found no matching records{scope} in the knowledge graph."
    if intent == "machines":
        return f"The assembly line has **{len(rows)} machines**:\n" + "\n".join(
            f"- **{r['name']}** ({r['id']}, {r['type']}) - {r['status']}" for r in rows)
    if intent == "status":
        return "Machine status:\n" + "\n".join(f"- **{r['name']}** ({r['id']}): {r['status']}" for r in rows)
    if intent == "overview":
        return "\n\n".join(
            f"**{r['name']}** ({r['id']}) - type {r['type']}, status **{r['status']}**\n"
            f"- Parts: {', '.join(sorted(r['parts'])) or 'none'}\n"
            f"- Sensors: {', '.join(sorted(r['sensors'])) or 'none'}\n"
            f"- Maintained by: {', '.join(sorted(r['technicians'])) or 'unassigned'}" for r in rows)
    if intent == "parts":
        by_m: dict[str, list[str]] = {}
        for r in rows:
            by_m.setdefault(f"{r['machine']} ({r['machine_id']})", []).append(r["part"])
        if pnames and not mids:
            return f"{', '.join(p.title() for p in pnames)} is used on:\n" + "\n".join(f"- **{m}**" for m in by_m)
        return "\n".join(f"**{m}** has {len(ps)} parts: {', '.join(ps)}" for m, ps in by_m.items())
    if intent == "sensors":
        return f"Sensors{scope}:\n" + "\n".join(
            f"- **{r['machine']}** {r['type']}: reading {r['reading']} {r['unit']} (threshold {r['threshold']} {r['unit']})"
            + (" - **ABOVE THRESHOLD**" if float(r["reading"]) >= float(r["threshold"]) else "")
            for r in rows)
    if intent == "tickets":
        return f"**{len(rows)} maintenance tickets**{scope} (newest first):\n" + "\n".join(
            f"- **{r['id']}** ({r['timestamp'][:10]}) {r['machine']} / {r['part']}: {r['text']}" for r in rows)
    if intent == "technician":
        return "Maintenance assignments:\n" + "\n".join(
            f"- **{r['technician']}** ({r['specialization']}) maintains **{r['machine']}** - last maintenance {r['last_maintenance']}"
            for r in rows)
    if intent == "at_risk":
        return ("Parts with **2 or more failure tickets** (predictive-maintenance watchlist):\n" + "\n".join(
            f"- **{r['part']}** on {r['machine']} - {r['ticket_count']} tickets ({', '.join(sorted(r['tickets']))})" for r in rows))
    return str(rows)


def detect_count_label(query: str) -> str:
    q = query.lower()
    for label, pattern in COUNT_LABELS:
        if re.search(pattern, q):
            return label
    return "Machine"


def answer(query: str, store, ents: Entities) -> dict:
    """Run the template pipeline. Returns answer text, the Cypher used, rows and intent."""
    import reasoning  # root cause, troubleshooting, impact and spares questions
    reasoned = reasoning.answer(query, store, ents.machines, ents.parts)
    if reasoned is not None:
        return reasoned
    machines = store.nodes("Machine")
    machines_by_id = {m["id"]: m for m in machines}
    intent = detect_intent(query, ents)
    label = detect_count_label(query) if intent == "count" else None
    cypher = COUNT_CYPHER[label] if intent == "count" else CYPHER[intent]
    params = {"mids": ents.machines, "pnames": ents.parts}

    if store.backend == "neo4j":
        rows = store.run_cypher(cypher, **params)
    else:
        rows = _python_rows(intent, store, ents.machines, ents.parts, label)
    text = format_answer(intent, rows, ents.machines, ents.parts, machines_by_id, label)
    return {"answer": text, "cypher": cypher, "params": params, "rows": rows, "intent": intent}
