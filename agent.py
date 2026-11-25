"""Cognitive engine: a LangGraph state machine.

    router_node ──GRAPH──▶ graph_qa_node  ──▶ END
                └─VECTOR─▶ vector_qa_node ──▶ END

* router_node     deterministic keyword routing; asks Mistral only when no rule fires.
* graph_qa_node   GraphCypherQAChain (text -> Cypher -> Neo4j -> answer) when Neo4j + Mistral are
                  up; otherwise the deterministic Cypher templates in graph_templates.py.
* vector_qa_node  k-NN search (k=2) over SOP embeddings, then a strict RAG prompt to Mistral;
                  without Mistral, returns the retrieved SOP text verbatim (no generation at all).
"""
from __future__ import annotations

import json
import re
import threading
from typing import List, Optional, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

import config
import graph_templates as gt
import reasoning
from embeddings import HASHING_NAME, get_embedder
from kg_store import UnsafeCypherError, assert_read_only


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    history: List[dict]
    route: str
    answer: str
    cypher: Optional[str]
    highlight: List[str]
    sources: List[dict]
    mode: str
    entities: dict


GRAPH_WORDS = re.compile(
    r"how many|count|number of|\blist\b|which (parts?|machines?|technician)|parts? of|components?|sensors?|threshold|"
    r"reading|tickets?|failures? (on|of|for)|history|who (maintains|is responsible)|technician|last maintenance|"
    r"when was|serviced|maintained|status|at[- ]risk|most (fail|ticket)|hierarchy|connected|tell me about|"
    r"overview|what parts"
)
VECTOR_WORDS = re.compile(
    r"how (do|to|can|should)|procedure|steps?|\bsops?\b|\bfix\b|repair|replace|what should|reset|"
    r"prime|priming|calibrat|safety|lockout|loto|ppe|instruction|guide|clean|align|tension|overheat|leak"
)

ROUTER_PROMPT = """You route questions for an industrial maintenance assistant.
GRAPH = questions about counts, lists, parts, sensors, tickets, technicians, status, or how assets are connected.
VECTOR = questions asking how to do something: repair steps, procedures, SOPs, safety instructions.
Question: {query}
Reply with exactly one word: GRAPH or VECTOR."""

CYPHER_PROMPT = PromptTemplate.from_template("""Task: Generate a Cypher statement to query the knowledge graph of a manufacturing assembly line.
Rules:
- Use only the node labels, relationship types and properties in the schema below.
- The statement must be READ-ONLY: only MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.
- Relationship directions: (Machine)-[:HAS_PART]->(Part), (Machine)-[:HAS_SENSOR]->(Sensor),
  (Ticket)-[:REPORTS_FAILURE_ON]->(Part), (SOP)-[:APPLIES_TO]->(Machine), (Technician)-[:MAINTAINED_BY]->(Machine).
- Compare names case-insensitively, e.g. WHERE toLower(m.name) CONTAINS 'press'.
- Return the `id` property of every node you return, plus a readable name. Never return `embedding`.
Machines on the line: {machines}
Schema:
{schema}
Return only the Cypher statement, with no explanation and no backticks.
Question: {question}""")

QA_PROMPT = PromptTemplate.from_template("""You are the maintenance knowledge system for a manufacturing assembly line.
Answer the question using ONLY the database results below. They are authoritative; do not add facts.
If the results are empty, say that no matching records were found.
Keep the answer short and use bullet points for lists.
Results: {context}
Question: {question}
Answer:""")

RAG_PROMPT = """You are the maintenance knowledge system for a manufacturing assembly line.
Answer ONLY from the SOP context below. Do not use outside knowledge and do not invent steps,
values or part numbers. If the context does not answer the question, reply exactly:
"No SOP in the knowledge base covers this. Please escalate to the shift maintenance supervisor."
Give the steps as a numbered list and cite the SOP id (e.g. SOP-CNC-01) you used.

SOP context:
{context}

Question: {query}
Answer:"""

STRUCTURED_SCHEMA = {
    "node_props": {
        "Machine": [{"property": p, "type": "STRING"} for p in ("id", "name", "type", "status", "tier")],
        "Part": [{"property": p, "type": "STRING"} for p in ("id", "name", "machine_id")],
        "Sensor": [{"property": "id", "type": "STRING"}, {"property": "type", "type": "STRING"},
                   {"property": "unit", "type": "STRING"}, {"property": "reading", "type": "FLOAT"},
                   {"property": "threshold", "type": "FLOAT"}],
        "Ticket": [{"property": p, "type": "STRING"} for p in ("id", "text", "timestamp", "machine_id")],
        "SOP": [{"property": p, "type": "STRING"} for p in ("id", "title", "text", "source")],
        "Technician": [{"property": p, "type": "STRING"} for p in ("id", "name", "specialization", "shift")],
    },
    "rel_props": {"MAINTAINED_BY": [{"property": "last_maintenance", "type": "STRING"}]},
    "relationships": [
        {"start": "Machine", "type": "HAS_PART", "end": "Part"},
        {"start": "Machine", "type": "HAS_SENSOR", "end": "Sensor"},
        {"start": "Ticket", "type": "REPORTS_FAILURE_ON", "end": "Part"},
        {"start": "SOP", "type": "APPLIES_TO", "end": "Machine"},
        {"start": "Technician", "type": "MAINTAINED_BY", "end": "Machine"},
    ],
    "metadata": {"constraint": [], "index": []},
}


class KGAgent:
    def __init__(self, store, llm_provider):
        self.store = store
        self.llm_provider = llm_provider
        self.embedder = get_embedder(store.embedder_name() or None)
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._cypher_chain = None
        self._refresh_catalog()
        self.app = self._build()

    # ------------------------------------------------------------------ setup
    def _refresh_catalog(self):
        self.machines = self.store.nodes("Machine")
        self.part_names = sorted({p["name"] for p in self.store.nodes("Part")})
        self.name_index = {}  # lower-case name/id -> node ids, used to highlight nodes from LLM output
        for label in ("Machine", "Part", "Technician", "Sensor", "Ticket", "SOP"):
            for n in self.store.nodes(label):
                self.name_index.setdefault(n["id"].lower(), set()).add(n["id"])
                if label in ("Machine", "Technician") and n.get("name"):
                    self.name_index.setdefault(n["name"].lower(), set()).add(n["id"])

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("router_node", self.router_node)
        g.add_node("graph_qa_node", self.graph_qa_node)
        g.add_node("vector_qa_node", self.vector_qa_node)
        g.set_entry_point("router_node")
        g.add_conditional_edges("router_node", lambda s: s["route"],
                                {"GRAPH": "graph_qa_node", "VECTOR": "vector_qa_node"})
        g.add_edge("graph_qa_node", END)
        g.add_edge("vector_qa_node", END)
        return g.compile()

    def _get_cypher_chain(self, llm):
        if self._cypher_chain is None:
            from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

            class ReadOnlyNeo4jGraph(Neo4jGraph):
                """Every Cypher statement the LLM generates passes the read-only guard first."""

                def query(self, query, params=None, session_params=None):
                    assert_read_only(query)
                    return super().query(query, params or {}, session_params or {})

            graph = ReadOnlyNeo4jGraph(url=config.NEO4J_URI, username=config.NEO4J_USER,
                                       password=config.NEO4J_PASSWORD, database=config.NEO4J_DATABASE,
                                       refresh_schema=False)  # schema is supplied below, no APOC needed
            graph.structured_schema = STRUCTURED_SCHEMA
            graph.schema = json.dumps(STRUCTURED_SCHEMA)
            machines = "; ".join(f"{m['id']} {m['name']}" for m in self.machines)
            self._cypher_chain = GraphCypherQAChain.from_llm(
                llm=llm, graph=graph,
                cypher_prompt=CYPHER_PROMPT.partial(machines=machines),
                qa_prompt=QA_PROMPT, validate_cypher=True, top_k=25,
                return_intermediate_steps=True, allow_dangerous_requests=True,
            )
        return self._cypher_chain

    # ------------------------------------------------------------------ helpers
    def _session(self, sid: str) -> dict:
        with self._lock:
            return self._sessions.setdefault(sid, {"history": [], "last": None})

    def _entities(self, state: AgentState) -> gt.Entities:
        last = self._session(state.get("session_id", "default"))["last"]
        return gt.resolve_entities(state["query"], self.machines, self.part_names, last)

    def _highlight_from_rows(self, rows: list[dict]) -> list[str]:
        ids = set()
        for r in rows:
            for k, v in r.items():
                if k in ("id", "machine_id", "part_id") and isinstance(v, str):
                    ids.add(v)
                elif k == "tickets" and isinstance(v, list):
                    ids.update(x for x in v if isinstance(x, str))
        return sorted(ids)

    def _highlight_from_text(self, *texts) -> list[str]:
        blob = " ".join(texts).lower()
        ids = set()
        for key, nids in self.name_index.items():
            if re.search(rf"(?<![a-z0-9-]){re.escape(key)}(?![a-z0-9-])", blob):
                ids.update(nids)
        return sorted(ids)

    # ------------------------------------------------------------------ nodes
    def router_node(self, state: AgentState) -> dict:
        q = state["query"].lower()
        g, v = len(GRAPH_WORDS.findall(q)), len(VECTOR_WORDS.findall(q))
        if reasoning.detect_intent(q):  # root cause / troubleshooting / impact / spares walk the graph
            route, how = "GRAPH", f"keyword rules ({reasoning.detect_intent(q).replace('_', ' ')})"
        elif g or v:
            route = "VECTOR" if v > g or (v == g and re.match(r"\s*how (do|to|can|should)", q)) else "GRAPH"
            how = "keyword rules"
        else:
            route, how = "VECTOR", "default"
            llm = self.llm_provider.get()
            if llm is not None:
                try:
                    out = llm.invoke(ROUTER_PROMPT.format(query=state["query"])).content.upper()
                    route, how = ("GRAPH" if "GRAPH" in out else "VECTOR"), "Mistral classifier"
                except Exception:  # noqa: BLE001
                    pass
        ents = self._entities(state)
        return {"route": route, "mode": f"routed by {how}",
                "entities": {"machines": ents.machines, "parts": ents.parts, "carried_over": ents.carried_over}}

    def graph_qa_node(self, state: AgentState) -> dict:
        ents = gt.Entities(**state["entities"])
        template = gt.answer(state["query"], self.store, ents)
        if template["intent"] in ("root_cause", "troubleshoot", "impact", "spares"):
            where = "Neo4j" if self.store.backend == "neo4j" else "in-memory graph"
            return {"answer": template["answer"], "cypher": template["cypher"], "highlight": template["highlight"],
                    "mode": f"graph reasoning ({template['intent'].replace('_', ' ')}) on {where}"}
        llm = self.llm_provider.get()

        if llm is not None and self.store.backend == "neo4j" and not ents.carried_over:
            try:
                res = self._get_cypher_chain(llm).invoke({"query": state["query"]})
                steps = res.get("intermediate_steps", [])
                cypher = steps[0].get("query") if steps else None
                context = steps[1].get("context", []) if len(steps) > 1 else []
                if context:  # trust the LLM path only when its Cypher actually returned data
                    hl = self._highlight_from_text(json.dumps(context, default=str), res["result"])
                    return {"answer": res["result"], "cypher": cypher, "highlight": hl,
                            "mode": "GraphCypherQAChain (Mistral -> Cypher -> Neo4j)"}
                note = "LLM Cypher returned no rows"
            except UnsafeCypherError as exc:
                note = f"blocked unsafe Cypher ({exc})"
            except Exception as exc:  # noqa: BLE001
                note = f"LLM Cypher failed ({exc.__class__.__name__})"
            mode = f"Cypher template on Neo4j ({note})"
        elif self.store.backend == "neo4j":
            mode = "Cypher template on Neo4j"
        else:
            mode = "Cypher template on in-memory graph"

        return {"answer": template["answer"], "cypher": template["cypher"],
                "highlight": self._highlight_from_rows(template["rows"]), "mode": mode}

    def vector_qa_node(self, state: AgentState) -> dict:
        ents = gt.Entities(**state["entities"])
        query_vec = self.embedder.encode([state["query"]])[0]
        k = config.VECTOR_TOP_K
        hits = self.store.vector_search(query_vec, k=k * 4 if ents.machines else k)
        if ents.machines:  # prefer SOPs that apply to the machine the user named
            wanted = set(ents.machines)
            hits.sort(key=lambda h: (not (set(h["machines"]) & wanted), -h["score"]))
        floor = 0.05 if self.embedder.name == HASHING_NAME else 0.25
        if not hits or max(h["score"] for h in hits) < floor:
            return {"answer": "No SOP in the knowledge base covers this. Please escalate to the shift maintenance supervisor.",
                    "sources": [], "highlight": [], "cypher": None, "mode": "vector search (no relevant SOP)"}
        top = max(h["score"] for h in hits)
        hits = [h for h in hits if h["score"] >= max(floor, 0.6 * top)][:k]  # drop weak runner-up matches
        sources = [{"id": h["id"], "title": h["title"], "source": h["source"], "score": round(h["score"], 3)} for h in hits]
        highlight = sorted({h["id"] for h in hits} | set(ents.machines))

        llm = self.llm_provider.get()
        if llm is not None:
            context = "\n\n".join(f"[{h['id']}] {h['title']}\n{h['text']}" for h in hits)
            try:
                text = llm.invoke(RAG_PROMPT.format(context=context, query=state["query"])).content.strip()
                return {"answer": text, "sources": sources, "highlight": highlight, "cypher": None,
                        "mode": f"vector RAG (k={k}) + Mistral"}
            except Exception:  # noqa: BLE001
                pass
        # Extractive answer: return the SOP text itself - zero hallucination risk.
        parts = []
        for h in hits:
            steps = re.sub(r"\s(?=\d+\.\s)", "\n", h["text"])  # put numbered steps on their own lines
            parts.append(f"**{h['title']}** (from {h['source']}, similarity {h['score']:.2f})\n{steps}")
        return {"answer": "\n\n".join(parts), "sources": sources, "highlight": highlight, "cypher": None,
                "mode": f"vector search (k={k}), extractive - Mistral offline"}

    # ------------------------------------------------------------------ public
    def ask(self, query: str, session_id: str = "default") -> dict:
        sess = self._session(session_id)
        state = self.app.invoke({"query": query, "session_id": session_id, "history": list(sess["history"])})
        ents = state.get("entities") or {}
        if ents.get("machines") or ents.get("parts"):
            sess["last"] = gt.Entities(machines=ents["machines"], parts=ents["parts"])
        sess["history"] = (sess["history"] + [{"q": query, "a": state.get("answer", "")}])[-10:]
        return {
            "answer": state.get("answer", ""),
            "route": state.get("route"),
            "cypher": state.get("cypher"),
            "highlight": state.get("highlight", []),
            "sources": state.get("sources", []),
            "mode": state.get("mode"),
            "entities": ents,
        }
