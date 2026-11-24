"""Storage layer: Neo4j property graph, with an in-memory fallback built from the JSON snapshot.

Both stores expose the same read API so the agent and API server don't care which is live:
    graph_data(), at_risk_parts(), vector_search(), nodes(label), edges(rel_type), run_cypher()
"""
from __future__ import annotations

import re

import numpy as np

import config
import graph_io

# Read-only guard for any Cypher produced by the LLM (protects against injection).
WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL|FOREACH|GRANT|REVOKE|DENY)\b",
    re.IGNORECASE,
)


class UnsafeCypherError(ValueError):
    pass


def assert_read_only(cypher: str) -> None:
    stripped = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)  # ignore keywords inside string literals
    m = WRITE_CLAUSES.search(stripped)
    if m:
        raise UnsafeCypherError(f"Blocked non read-only Cypher clause: {m.group(1).upper()}")


def _flat(node: dict) -> dict:
    return {"id": node["id"], "label": node["label"], **node["props"]}


class MemoryStore:
    """Reads data/graph_snapshot.json. Used when Neo4j isn't running (demo-safe fallback)."""

    backend = "memory"

    def __init__(self):
        snap = graph_io.load_snapshot()
        if not snap["nodes"]:
            raise RuntimeError("Knowledge graph is empty - run `python ingest.py` and `python ingest_pdf.py`.")
        self.meta = snap.get("meta", {})
        self._nodes = {n["id"]: n for n in snap["nodes"]}
        self._edges = snap["edges"]
        sops = [n for n in snap["nodes"] if n["label"] == "SOP" and n["props"].get("embedding")]
        self._sop_ids = [n["id"] for n in sops]
        self._sop_matrix = np.asarray([n["props"]["embedding"] for n in sops], dtype=np.float32) if sops else None

    # -- generic accessors ---------------------------------------------------
    def node(self, node_id: str) -> dict | None:
        n = self._nodes.get(node_id)
        return _flat(n) if n else None

    def nodes(self, label: str) -> list[dict]:
        return [_flat(n) for n in self._nodes.values() if n["label"] == label]

    def edges(self, rel_type: str) -> list[dict]:
        return [e for e in self._edges if e["type"] == rel_type]

    def embedder_name(self) -> str | None:
        return self.meta.get("embedder")

    def run_cypher(self, cypher: str, **params):
        raise NotImplementedError("Cypher needs Neo4j; the memory store answers via templates.")

    # -- API helpers -----------------------------------------------------------
    def graph_data(self) -> dict:
        nodes = []
        for n in self._nodes.values():
            props = {k: v for k, v in n["props"].items() if k != "embedding"}  # strip 384-d vectors
            nodes.append({"id": n["id"], "label": n["label"], "props": props})
        links = [{"source": e["source"], "target": e["target"], "type": e["type"], "props": e.get("props", {})}
                 for e in self._edges]
        return {"nodes": nodes, "links": links}

    def at_risk_parts(self, min_tickets: int = config.AT_RISK_MIN_TICKETS) -> list[dict]:
        tickets_by_part: dict[str, list[str]] = {}
        for e in self.edges("REPORTS_FAILURE_ON"):
            tickets_by_part.setdefault(e["target"], []).append(e["source"])
        rows = []
        for pid, tks in tickets_by_part.items():
            if len(tks) >= min_tickets:
                part, machine = self.node(pid), self.node(self.node(pid)["machine_id"])
                latest = max(self.node(t)["timestamp"] for t in tks)
                rows.append({"part_id": pid, "part": part["name"], "machine_id": machine["id"],
                             "machine": machine["name"], "ticket_count": len(tks),
                             "tickets": sorted(tks), "last_failure": latest})
        return sorted(rows, key=lambda r: (-r["ticket_count"], r["part_id"]))

    def vector_search(self, embedding: np.ndarray, k: int = config.VECTOR_TOP_K) -> list[dict]:
        if self._sop_matrix is None:
            return []
        scores = self._sop_matrix @ np.asarray(embedding, dtype=np.float32)  # vectors are L2-normalised
        out = []
        for idx in np.argsort(-scores)[:k]:
            sop = self.node(self._sop_ids[idx])
            machines = [e["target"] for e in self.edges("APPLIES_TO") if e["source"] == sop["id"]]
            out.append({"id": sop["id"], "title": sop["title"], "text": sop["text"], "source": sop["source"],
                        "score": round(float(scores[idx]), 4), "machines": machines})
        return out


class Neo4jStore:
    backend = "neo4j"

    def __init__(self, driver):
        self.driver = driver

    def _read(self, cypher: str, **params) -> list[dict]:
        with self.driver.session(database=config.NEO4J_DATABASE) as s:
            return s.execute_read(lambda tx: [r.data() for r in tx.run(cypher, params)])

    def run_cypher(self, cypher: str, **params) -> list[dict]:
        assert_read_only(cypher)
        return self._read(cypher, **params)

    def node(self, node_id: str) -> dict | None:
        rows = self._read("MATCH (n {id: $id}) RETURN labels(n)[0] AS label, properties(n) AS p", id=node_id)
        if not rows:
            return None
        p = {k: v for k, v in rows[0]["p"].items() if k != "embedding"}
        return {"label": rows[0]["label"], **p}

    def nodes(self, label: str) -> list[dict]:
        assert label in graph_io.LABELS
        rows = self._read(f"MATCH (n:{label}) RETURN properties(n) AS p")
        return [{"label": label, **{k: v for k, v in r["p"].items() if k != "embedding"}} for r in rows]

    def edges(self, rel_type: str) -> list[dict]:
        assert rel_type.isupper()
        return self._read(f"MATCH (a)-[r:{rel_type}]->(b) RETURN a.id AS source, b.id AS target, "
                          f"type(r) AS type, properties(r) AS props")

    def embedder_name(self) -> str | None:
        rows = self._read("MATCH (s:SOP) RETURN s.embedder AS e LIMIT 1")
        return rows[0]["e"] if rows else None

    def graph_data(self) -> dict:
        nodes = self._read("MATCH (n) RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS props")
        for n in nodes:
            n["props"].pop("embedding", None)  # strip 384-d vectors from the payload
        links = self._read("MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, "
                           "type(r) AS type, properties(r) AS props")
        return {"nodes": nodes, "links": links}

    def at_risk_parts(self, min_tickets: int = config.AT_RISK_MIN_TICKETS) -> list[dict]:
        return self._read(
            "MATCH (m:Machine)-[:HAS_PART]->(p:Part)<-[:REPORTS_FAILURE_ON]-(t:Ticket) "
            "WITH m, p, count(t) AS c, collect(t.id) AS tks, max(t.timestamp) AS last "
            "WHERE c >= $min "
            "RETURN p.id AS part_id, p.name AS part, m.id AS machine_id, m.name AS machine, "
            "c AS ticket_count, tks AS tickets, last AS last_failure ORDER BY c DESC, p.id",
            min=min_tickets,
        )

    def vector_search(self, embedding: np.ndarray, k: int = config.VECTOR_TOP_K) -> list[dict]:
        return self._read(
            f"CALL db.index.vector.queryNodes('{config.VECTOR_INDEX}', $k, $emb) YIELD node, score "
            "OPTIONAL MATCH (node)-[:APPLIES_TO]->(m:Machine) "
            "RETURN node.id AS id, node.title AS title, node.text AS text, node.source AS source, "
            "score, collect(m.id) AS machines ORDER BY score DESC",
            k=k, emb=[float(x) for x in embedding],
        )


def get_store():
    """Neo4j when configured and reachable, otherwise the in-memory snapshot."""
    driver = graph_io.connect_neo4j()
    if driver is not None:
        return Neo4jStore(driver)
    return MemoryStore()
