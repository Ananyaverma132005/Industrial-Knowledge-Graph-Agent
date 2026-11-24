"""Data ingestion.

Parses the asset registry and maintenance-ticket JSON, runs the spaCy NLP pipeline with a
custom entity_ruler that tags machine parts as PART entities, and commits the resulting
graph to Neo4j (MERGE) and to the local JSON snapshot.

    python ingest.py
"""
import json

import spacy

import config
import graph_io

STRUCTURAL_LABELS = {"Machine", "Part", "Sensor", "Ticket", "Technician"}


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


def build_graph(registry: dict, tickets: list[dict], nlp) -> tuple[list, list, list]:
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
    return nodes, edges, warnings


def main() -> None:
    registry = json.loads(config.ASSET_REGISTRY.read_text(encoding="utf-8"))
    tickets = json.loads(config.TICKETS_FILE.read_text(encoding="utf-8"))
    all_parts = [p for m in registry["machines"] for p in m["parts"]]
    nlp, model = build_nlp(all_parts)
    print(f"[nlp] spaCy pipeline: {model} + entity_ruler ({len(set(all_parts))} PART patterns)")

    nodes, edges, warnings = build_graph(registry, tickets, nlp)
    for w in warnings:
        print("[nlp] warning:", w)
    links = sum(1 for e in edges if e["type"] == "REPORTS_FAILURE_ON")
    print(f"[graph] {len(nodes)} nodes, {len(edges)} relationships ({links} REPORTS_FAILURE_ON from NLP)")

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
