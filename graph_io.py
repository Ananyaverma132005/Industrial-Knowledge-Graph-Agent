"""Shared helpers for writing the knowledge graph to Neo4j and to the local JSON snapshot.

The snapshot is what the in-memory fallback store reads when Neo4j is not running,
so every ingestion script writes to both.
"""
import json
import time
from collections import defaultdict

import config

LABELS = ["Machine", "Part", "Sensor", "Ticket", "SOP", "Technician", "FailureMode", "SparePart"]


def load_snapshot() -> dict:
    if config.SNAPSHOT_FILE.exists():
        return json.loads(config.SNAPSHOT_FILE.read_text(encoding="utf-8"))
    return {"meta": {}, "nodes": [], "edges": []}


def save_snapshot(snapshot: dict) -> None:
    snapshot.setdefault("meta", {})["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config.SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.SNAPSHOT_FILE.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")


def replace_in_snapshot(snapshot: dict, nodes: list, edges: list, labels: set) -> dict:
    """Replace all nodes with one of `labels` and their *outgoing* edges.

    Edges owned by other ingestion steps (e.g. SOP -[:APPLIES_TO]-> Machine written by
    ingest_pdf.py) survive as long as both endpoints still exist.
    """
    old_label = {n["id"]: n["label"] for n in snapshot["nodes"]}
    kept_nodes = [n for n in snapshot["nodes"] if n["label"] not in labels]
    snapshot["nodes"] = kept_nodes + nodes
    alive = {n["id"] for n in snapshot["nodes"]}
    snapshot["edges"] = [
        e for e in snapshot["edges"]
        if old_label.get(e["source"]) not in labels and e["source"] in alive and e["target"] in alive
    ] + edges
    return snapshot


def connect_neo4j():
    """Return a verified Neo4j driver, or None if Neo4j is disabled/unreachable."""
    if config.USE_NEO4J == "false" or not config.NEO4J_PASSWORD:
        return None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        print(f"[neo4j] not available ({exc.__class__.__name__}: {exc}); using JSON snapshot only")
        return None


def push_to_neo4j(driver, nodes: list, edges: list, delete_labels: set) -> None:
    """MERGE nodes and relationships. Labels/types come from our own whitelist, never user input."""
    with driver.session(database=config.NEO4J_DATABASE) as s:
        for label in LABELS:
            s.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")
        # Same rule as replace_in_snapshot: drop nodes that no longer exist and the
        # outgoing edges of the re-ingested labels; keep edges owned by other steps.
        for label in delete_labels:
            assert label in LABELS
            ids = [n["id"] for n in nodes if n["label"] == label]
            s.run(f"MATCH (n:{label}) WHERE NOT n.id IN $ids DETACH DELETE n", ids=ids)
            s.run(f"MATCH (n:{label})-[r]->() DELETE r")

        by_label = defaultdict(list)
        for n in nodes:
            by_label[n["label"]].append({"id": n["id"], "props": n["props"]})
        for label, rows in by_label.items():
            assert label in LABELS
            s.run(f"UNWIND $rows AS r MERGE (n:{label} {{id: r.id}}) SET n += r.props", rows=rows)

        label_of = {n["id"]: n["label"] for n in nodes}
        # Endpoints may already exist in Neo4j from an earlier ingest (e.g. SOP -> Machine).
        missing = {e["target"] for e in edges if e["target"] not in label_of} | {
            e["source"] for e in edges if e["source"] not in label_of
        }
        if missing:
            for rec in s.run("MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, labels(n)[0] AS label", ids=list(missing)):
                label_of[rec["id"]] = rec["label"]

        by_type = defaultdict(list)
        for e in edges:
            key = (e["type"], label_of[e["source"]], label_of[e["target"]])
            by_type[key].append({"s": e["source"], "t": e["target"], "props": e.get("props", {})})
        for (rtype, sl, tl), rows in by_type.items():
            assert sl in LABELS and tl in LABELS and rtype.isupper()
            s.run(
                f"UNWIND $rows AS r MATCH (a:{sl} {{id: r.s}}) MATCH (b:{tl} {{id: r.t}}) "
                f"MERGE (a)-[e:{rtype}]->(b) SET e += r.props",
                rows=rows,
            )
