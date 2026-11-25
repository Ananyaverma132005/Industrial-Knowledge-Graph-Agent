"""Data ingestion.

Parses the asset registry, maintenance tickets and the failure-mode / spare-part knowledge
base. spaCy tags machine parts in ticket text (entity_ruler, PART entities) and finds failure
symptoms and causes (PhraseMatcher), so each ticket is linked to the part it reports and the
failure modes it indicates. The graph is committed to Neo4j (MERGE) and to the JSON snapshot.

    python ingest.py
"""
import json

import spacy
from spacy.matcher import PhraseMatcher

import config
import graph_io

STRUCTURAL_LABELS = {"Machine", "Part", "Sensor", "Ticket", "Technician", "FailureMode", "SparePart"}


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.upper()).strip("-")


def build_nlp(part_names: list[str]):
    """en_core_web_sm + entity_ruler injected *before* NER."""
    try:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        model = "en_core_web_sm"
    except OSError:
        # Model not downloaded: a blank English pipeline still runs the rule-based matcher.
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        model = "blank:en"
    patterns = [
        {"label": "PART", "pattern": [{"LOWER": tok} for tok in name.lower().split()], "id": name}
        for name in sorted(set(part_names))
    ]
    ruler.add_patterns(patterns)
    return nlp, model


def extract_parts(nlp, text: str) -> list[str]:
    """Return canonical part names (the ruler's pattern id) mentioned in the text."""
    doc = nlp(text)
    return sorted({ent.ent_id_ or ent.text.title() for ent in doc.ents if ent.label_ == "PART"})


def build_phrase_matcher(nlp, kb: dict) -> PhraseMatcher:
    """Failure-mode phrases (FM ids) and spare-part names (SP ids), matched case-insensitively."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for fm in kb["failure_modes"]:
        matcher.add(fm["id"], [nlp.make_doc(p) for p in fm["phrases"]])
    for sp in kb["spare_parts"]:
        matcher.add(sp["id"], [nlp.make_doc(sp["name"])])
    return matcher


def match_ids(nlp, matcher, text: str, prefix: str) -> list[str]:
    doc = nlp.make_doc(text)
    return sorted({nlp.vocab.strings[mid] for mid, _, _ in matcher(doc) if nlp.vocab.strings[mid].startswith(prefix)})


def build_graph(registry: dict, tickets: list[dict], nlp, kb: dict | None = None) -> tuple[list, list, list]:
    nodes, edges, warnings = [], [], []
    parts_by_machine: dict[str, dict[str, str]] = {}

    for m in registry["machines"]:
        nodes.append({"id": m["id"], "label": "Machine", "props": {
            "id": m["id"], "name": m["name"], "type": m["type"], "status": m["status"], "tier": m["tier"]}})
        parts_by_machine[m["id"]] = {}
        for pname in m["parts"]:
            pid = f"{m['id']}-{slug(pname)}"
            parts_by_machine[m["id"]][pname.lower()] = pid
            nodes.append({"id": pid, "label": "Part", "props": {"id": pid, "name": pname, "machine_id": m["id"]}})
            edges.append({"source": m["id"], "target": pid, "type": "HAS_PART", "props": {}})
        for s in m["sensors"]:
            sid = f"{m['id']}-{slug(s['type'])}"
            nodes.append({"id": sid, "label": "Sensor", "props": {
                "id": sid, "type": s["type"], "unit": s["unit"], "reading": s["reading"], "threshold": s["threshold"]}})
            edges.append({"source": m["id"], "target": sid, "type": "HAS_SENSOR", "props": {}})

    for t in registry["technicians"]:
        nodes.append({"id": t["id"], "label": "Technician", "props": {
            "id": t["id"], "name": t["name"], "specialization": t["specialization"], "shift": t["shift"]}})
        for link in t["maintains"]:
            # Relationship direction: (Technician)-[:MAINTAINED_BY]->(Machine)
            edges.append({"source": t["id"], "target": link["machine_id"], "type": "MAINTAINED_BY",
                          "props": {"last_maintenance": link["last_maintenance"]}})

    for tk in tickets:
        nodes.append({"id": tk["id"], "label": "Ticket", "props": {
            "id": tk["id"], "text": tk["text"], "timestamp": tk["timestamp"], "machine_id": tk["machine_id"]}})
        found = extract_parts(nlp, tk["text"])
        machine_parts = parts_by_machine.get(tk["machine_id"], {})
        linked = [machine_parts[p.lower()] for p in found if p.lower() in machine_parts]
        if not linked:
            warnings.append(f"{tk['id']}: no part of {tk['machine_id']} found in text (entities: {found})")
        for pid in linked:
            edges.append({"source": tk["id"], "target": pid, "type": "REPORTS_FAILURE_ON", "props": {}})

    if kb:
        add_failure_knowledge(registry, tickets, nlp, kb, parts_by_machine, nodes, edges)
    return nodes, edges, warnings


def add_failure_knowledge(registry, tickets, nlp, kb, parts_by_machine, nodes, edges) -> None:
    """Failure modes, spare parts, production flow and the ticket links that reasoning relies on."""
    # Production order: (Machine)-[:FEEDS]->(Machine)
    for src, dst in registry.get("line_flow", []):
        edges.append({"source": src, "target": dst, "type": "FEEDS", "props": {}})

    for sp in kb["spare_parts"]:
        nodes.append({"id": sp["id"], "label": "SparePart", "props": {
            "id": sp["id"], "name": sp["name"], "stock": sp["stock"],
            "lead_time_days": sp["lead_time_days"], "location": sp["location"]}})
        wanted = {p.lower() for p in sp["for_parts"]}
        for mid, parts in parts_by_machine.items():
            for pname, pid in parts.items():
                if pname in wanted:
                    edges.append({"source": pid, "target": sp["id"], "type": "USES_SPARE", "props": {}})

    for fm in kb["failure_modes"]:
        nodes.append({"id": fm["id"], "label": "FailureMode", "props": {"id": fm["id"], "name": fm["name"]}})
        for cause in fm["causes"]:
            edges.append({"source": fm["id"], "target": cause, "type": "CAUSED_BY", "props": {}})
        for sop in fm["sops"]:
            edges.append({"source": fm["id"], "target": sop, "type": "RESOLVED_BY", "props": {}})
        for sp in fm["spares"]:
            edges.append({"source": fm["id"], "target": sp, "type": "NEEDS_SPARE", "props": {}})
        wanted = {p.lower() for p in fm["parts"]}
        for parts in parts_by_machine.values():
            for pname, pid in parts.items():
                if pname in wanted:
                    edges.append({"source": fm["id"], "target": pid, "type": "AFFECTS", "props": {}})

    matcher = build_phrase_matcher(nlp, kb)
    for tk in tickets:
        for fm_id in match_ids(nlp, matcher, tk["text"], "FM-"):
            edges.append({"source": tk["id"], "target": fm_id, "type": "INDICATES", "props": {}})
        if "replac" in tk["text"].lower():
            for sp_id in match_ids(nlp, matcher, tk["text"], "SP-"):
                edges.append({"source": tk["id"], "target": sp_id, "type": "USED_SPARE", "props": {}})


def main() -> None:
    registry = json.loads(config.ASSET_REGISTRY.read_text(encoding="utf-8"))
    tickets = json.loads(config.TICKETS_FILE.read_text(encoding="utf-8"))
    kb = json.loads(config.FAILURE_KB.read_text(encoding="utf-8"))
    all_parts = [p for m in registry["machines"] for p in m["parts"]]
    nlp, model = build_nlp(all_parts)
    print(f"[nlp] spaCy pipeline: {model} + entity_ruler ({len(set(all_parts))} PART patterns)")

    nodes, edges, warnings = build_graph(registry, tickets, nlp, kb)
    for w in warnings:
        print("[nlp] warning:", w)
    links = sum(1 for e in edges if e["type"] == "REPORTS_FAILURE_ON")
    indicates = sum(1 for e in edges if e["type"] == "INDICATES")
    print(f"[graph] {len(nodes)} nodes, {len(edges)} relationships "
          f"({links} REPORTS_FAILURE_ON and {indicates} INDICATES links from NLP)")

    snapshot = graph_io.replace_in_snapshot(graph_io.load_snapshot(), nodes, edges, STRUCTURAL_LABELS)
    graph_io.save_snapshot(snapshot)
    print(f"[snapshot] wrote {config.SNAPSHOT_FILE.relative_to(config.BASE_DIR)}")

    driver = graph_io.connect_neo4j()
    if driver:
        graph_io.push_to_neo4j(driver, nodes, edges, delete_labels=STRUCTURAL_LABELS)
        driver.close()
        print(f"[neo4j] committed graph to {config.NEO4J_URI}")


if __name__ == "__main__":
    main()
