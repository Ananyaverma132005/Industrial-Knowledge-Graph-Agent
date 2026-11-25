"""PDF vectorizer.

Reads PDF manuals, chunks them into SOP sections, encodes each chunk with
SentenceTransformers (all-MiniLM-L6-v2, 384-d) and stores them as SOP nodes with an
`embedding` property, linked (SOP)-[:APPLIES_TO]->(Machine). In Neo4j it also creates the
cosine vector index "sop_embeddings" used for k-NN search.

    python ingest_pdf.py            # all PDFs in data/manuals
"""
import re

from pypdf import PdfReader

import config
import graph_io
from embeddings import get_embedder

HEADING = re.compile(r"^(SOP-[A-Z]+-\d+)\s+(.*)$")
MAX_CHUNK = 1200


def read_pdf(path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def machines_for(stem: str, all_ids: list[str]) -> list[str]:
    """M1_x.pdf -> [M1]; M2_M6_x.pdf -> [M2, M6]; ALL_x.pdf -> all machines."""
    tokens = stem.split("_")
    if tokens[0].upper() == "ALL":
        return all_ids
    return [t.upper() for t in tokens if t.upper() in all_ids]


def chunk(text: str, source: str) -> list[dict]:
    """Split on 'SOP-XXX-NN Title' headings; fall back to fixed windows for unstructured PDFs."""
    chunks, current = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = HEADING.match(line)
        if m:
            current = {"sop_code": m.group(1), "title": f"{m.group(1)} {m.group(2)}", "lines": []}
            chunks.append(current)
        elif current is not None:
            current["lines"].append(line)
    if chunks:
        return [{"sop_code": c["sop_code"], "title": c["title"], "text": " ".join(c["lines"])} for c in chunks]
    # Unstructured manual: overlapping windows
    flat = " ".join(text.split())
    out, i, n = [], 0, 1
    while i < len(flat):
        out.append({"sop_code": f"{source}-{n}", "title": f"{source} part {n}", "text": flat[i:i + MAX_CHUNK]})
        i += MAX_CHUNK - 200
        n += 1
    return out


def main() -> None:
    snapshot = graph_io.load_snapshot()
    machine_ids = sorted(n["id"] for n in snapshot["nodes"] if n["label"] == "Machine")
    if not machine_ids:
        raise SystemExit("No machines in the graph yet - run `python ingest.py` first.")

    pdfs = sorted(config.MANUALS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs in data/manuals - run `python make_sample_manuals.py` or add manuals.")

    records = []
    # Public-domain manuals saved by scrape_manuals.py (plain text, first line is the title)
    for txt in sorted((config.MANUALS_DIR / "public").glob("*.txt")):
        targets = machines_for(txt.stem, machine_ids)
        lines = txt.read_text(encoding="utf-8").splitlines()
        title, body = lines[0], " ".join(lines[2:])
        code = "PUB-" + txt.stem.split("_osha_")[-1].upper().replace("_", "-")
        pieces = chunk(body, code)
        for n, p in enumerate(pieces, 1):
            p.update(sop_code=f"{code}-{n}", title=f"{title} (part {n})")
            records.append({**p, "source": f"{txt.name} (scraped, public domain)", "machines": targets})
        print(f"[txt] {txt.name}: {len(pieces)} chunks -> applies to {targets}")

    for pdf in pdfs:
        targets = machines_for(pdf.stem, machine_ids)
        pieces = chunk(read_pdf(pdf), pdf.stem)
        print(f"[pdf] {pdf.name}: {len(pieces)} chunks -> applies to {targets or 'no machine (check file name prefix)'}")
        for p in pieces:
            records.append({**p, "source": pdf.name, "machines": targets})

    embedder = get_embedder()
    vectors = embedder.encode([f"{r['title']}. {r['text']}" for r in records])
    print(f"[vectors] encoded {len(records)} chunks with {embedder.name} -> dim {vectors.shape[1]}")

    nodes, edges = [], []
    for r, vec in zip(records, vectors):
        sid = r["sop_code"]
        nodes.append({"id": sid, "label": "SOP", "props": {
            "id": sid, "title": r["title"], "text": r["text"], "source": r["source"],
            "embedder": embedder.name, "embedding": [round(float(x), 6) for x in vec]}})
        for mid in r["machines"]:
            edges.append({"source": sid, "target": mid, "type": "APPLIES_TO", "props": {}})

    snapshot = graph_io.replace_in_snapshot(snapshot, nodes, edges, {"SOP"})
    snapshot["meta"]["embedder"] = embedder.name
    graph_io.save_snapshot(snapshot)
    print(f"[snapshot] wrote {len(nodes)} SOP nodes to {config.SNAPSHOT_FILE.relative_to(config.BASE_DIR)}")

    driver = graph_io.connect_neo4j()
    if driver:
        graph_io.push_to_neo4j(driver, nodes, edges, delete_labels={"SOP"})
        with driver.session(database=config.NEO4J_DATABASE) as s:
            s.run(f"DROP INDEX {config.VECTOR_INDEX} IF EXISTS")
            dim = int(vectors.shape[1])
            s.run(
                f"CREATE VECTOR INDEX {config.VECTOR_INDEX} IF NOT EXISTS FOR (s:SOP) ON (s.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
            )
        driver.close()
        print(f"[neo4j] SOP nodes + vector index '{config.VECTOR_INDEX}' committed")


if __name__ == "__main__":
    main()
