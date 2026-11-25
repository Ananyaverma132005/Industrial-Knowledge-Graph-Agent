# Industrial Knowledge Graph Agent

A maintenance assistant for a factory assembly line. It links machines, parts, failures, spare parts, SOPs, maintenance tickets, manuals and the asset registry in one knowledge graph, and answers plain-English questions over it: root cause tracing, troubleshooting paths and impact analysis. All data is sample or public data.

## What it can answer

| Question type | Example |
| --- | --- |
| Root cause tracing | *What is the root cause of spindle overheating?* |
| Troubleshooting path | *Troubleshooting path for servo motor overcurrent* |
| Impact analysis | *What is the impact if the belt conveyor fails?* |
| Spare parts | *Which spare parts are out of stock?* |
| Asset questions | *Which parts are at risk of failure?*, *Who maintains the hydraulic press?* |
| Manuals (RAG) | *How do I fix a hydraulic seal leak?*, *How should belts and pulleys be guarded?* |

## Data sources

- **Asset registry**: 6 machines, 25 parts, 13 sensors, 4 technicians, and the order machines feed each other on the line
- **Maintenance tickets**: 24 sample tickets (Jan to Jul 2026)
- **Failure knowledge**: 29 failure modes with their causes, fixing SOPs and spare parts; 11 spare parts with stock and lead time
- **Manuals**: sample SOP manuals for each machine, plus public-domain OSHA manuals scraped by `scrape_manuals.py`
- **Sensor data**: the AI4I 2020 Predictive Maintenance dataset (UCI / Kaggle), used to train the failure-risk model

## How it works

1. `ingest.py` reads the registry and tickets, uses spaCy to find parts and failure modes in ticket text, and builds the graph
2. `ingest_pdf.py` splits the manuals into sections and stores their embeddings (MiniLM) for semantic search
3. A LangGraph agent routes each question: graph questions run Cypher on Neo4j, procedure questions use RAG over the manuals
4. `reasoning.py` walks the graph for root causes (`CAUSED_BY`, ranked by ticket evidence), troubleshooting (`RESOLVED_BY`, `NEEDS_SPARE`) and impact (`FEEDS`)

Graph: 151 nodes (Machine, Part, Sensor, Ticket, SOP, Technician, FailureMode, SparePart) and 342 relationships.

## Tech stack

Python, Neo4j, LangChain, LangGraph, Mistral (Ollama), SentenceTransformers, spaCy, scikit-learn, FastAPI, D3.js, Docker

## Run locally

Needs Python 3.11.

```powershell
git clone https://github.com/Ananyaverma132005/Industrial-Knowledge-Graph-Agent.git
cd Industrial-Knowledge-Graph-Agent
powershell -ExecutionPolicy Bypass -File setup.ps1
powershell -File run.ps1
```

Open http://localhost:8000. Neo4j and Ollama (Mistral) are optional. Without them the app uses a built-in graph and quotes answers directly from the manuals.

With Docker:

```bash
docker build -t kg-agent .
docker run -p 7860:7860 kg-agent
```

## Live demo

https://ananyaverma132005.github.io/Industrial-Knowledge-Graph-Agent/
