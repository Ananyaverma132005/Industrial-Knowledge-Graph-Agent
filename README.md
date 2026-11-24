# Industrial Knowledge Graph Agent

A demonstration project: a maintenance assistant for a factory assembly line. It stores machines, parts, sensors, technicians, repair tickets and safety procedures in a knowledge graph, and answers plain-English questions about them. All data is sample data.

## Features

- Ask questions like *"Which parts are at risk of failure?"* or *"How do I fix spindle overheating?"*
- Graph questions are answered with Cypher queries, and procedure questions by searching the manuals
- Interactive asset graph (D3.js) that highlights the parts an answer refers to
- Live machine telemetry with a failure-risk score from a Random Forest model
- Remembers context, so follow-ups like *"When was it last serviced?"* work

## Tech stack

Python, FastAPI, LangGraph, Neo4j, SentenceTransformers, spaCy, scikit-learn, D3.js, Docker

## Run locally

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
