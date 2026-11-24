"""API server.

Initialises FastAPI, configures the Neo4j driver (or the in-memory fallback), compiles the
LangGraph state machine and mounts the WebSocket telemetry endpoint + static client.

    uvicorn main:app --port 8000        then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from agent import KGAgent
from kg_store import get_store
from llm import LLMProvider
from telemetry import FailureModel, TelemetrySimulator

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_store()
    llm = LLMProvider()
    llm.get()  # first availability check, so /health is accurate immediately
    model = FailureModel()
    state.update(store=store, llm=llm, model=model, agent=KGAgent(store, llm))
    print(f"[startup] graph backend: {store.backend} | LLM: {llm.status} | "
          f"embedder: {state['agent'].embedder.name} | ML model: {model.source}")
    yield
    if store.backend == "neo4j":
        store.driver.close()


app = FastAPI(title="Industrial Knowledge Graph Agent", version="1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


class FailureFeatures(BaseModel):
    type: str = Field("M", pattern="^[LMH]$", description="Machine tier L/M/H")
    air_temperature_k: float = 300.0
    process_temperature_k: float = 310.0
    rotational_speed_rpm: float = 1500
    torque_nm: float = 40
    tool_wear_min: float = 100


@app.post("/chat")
async def chat(req: ChatRequest):
    """Invokes the LangGraph agent. Returns answer, generated Cypher and the routing path."""
    sid = req.session_id or str(uuid.uuid4())
    result = await run_in_threadpool(state["agent"].ask, req.query.strip(), sid)
    return {**result, "session_id": sid}


@app.get("/graph-data")
async def graph_data():
    """Node + link lists for D3. 384-d embedding vectors are stripped from the payload."""
    return await run_in_threadpool(state["store"].graph_data)


@app.get("/predict-maintenance")
async def predict_maintenance():
    """Parts with >= 2 failure tickets (Cypher aggregation) - the at-risk component list."""
    return await run_in_threadpool(state["store"].at_risk_parts, config.AT_RISK_MIN_TICKETS)


@app.post("/predict-failure")
async def predict_failure(f: FailureFeatures):
    """Scores one feature vector with the RandomForest model."""
    row = {"Type": f.type, "Air temperature [K]": f.air_temperature_k,
           "Process temperature [K]": f.process_temperature_k, "Rotational speed [rpm]": f.rotational_speed_rpm,
           "Torque [Nm]": f.torque_nm, "Tool wear [min]": f.tool_wear_min}
    p = state["model"].predict_proba([row])[0]
    return {"failure_probability": p, "prediction": "FAILURE LIKELY" if p >= 0.5 else "normal"}


@app.get("/node/{node_id}")
async def node(node_id: str):
    n = await run_in_threadpool(state["store"].node, node_id)
    if n is None:
        raise HTTPException(404, "node not found")
    return n


@app.get("/health")
async def health():
    llm = state["llm"]
    llm.get()
    return {"graph_backend": state["store"].backend, "llm": llm.status, "llm_ready": llm.get() is not None,
            "embedder": state["agent"].embedder.name, "ml_model": state["model"].source}


@app.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket):
    """Streams simulated thermal/vibration metrics + ML failure probability every 3 seconds."""
    await ws.accept()
    sim = TelemetrySimulator(state["store"].nodes("Machine"), state["model"])
    try:
        while True:
            await ws.send_json(await run_in_threadpool(sim.tick))
            await asyncio.sleep(config.TELEMETRY_INTERVAL)
    except (WebSocketDisconnect, RuntimeError):
        pass


# Static client last, so API routes take precedence.
app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
