/* Client app: DOM events, D3 force-directed graph, WebSocket telemetry. */

const COLORS = {
  Machine: "var(--machine)", Part: "var(--part)", Sensor: "var(--sensor)",
  Ticket: "var(--ticket)", SOP: "var(--sop)", Technician: "var(--technician)",
  FailureMode: "var(--failure)", SparePart: "var(--spare)",
};
const RADIUS = { Machine: 15, Technician: 10, Part: 7, Sensor: 6, Ticket: 5.5, SOP: 6.5, FailureMode: 7.5, SparePart: 6.5 };
const SUGGESTIONS = [
  ["What is the root cause of spindle overheating?", "Root cause"],
  ["Troubleshooting path for servo motor overcurrent", "Troubleshoot"],
  ["What is the impact if the belt conveyor fails?", "Impact"],
  ["Which spare parts are out of stock?", "Spares"],
  ["Which parts are at risk of failure?", "Graph"],
  ["Who maintains the hydraulic press?", "Graph"],
  ["How do I fix a hydraulic seal leak?", "Manual"],
  ["How should belts and pulleys be guarded?", "Manual"],
];
const SOURCE_LABEL = { GRAPH: "Knowledge graph", VECTOR: "SOP manuals" };

let sessionId = null;
let graph = { nodes: [], links: [] };
let nodeSel, linkSel, simulation, svg, zoom;
let lastHighlight = [];
const hiddenLabels = new Set();
const alerting = new Set();

/* ------------------------------------------------------------------ utils */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const timeNow = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** Small, safe markdown: escapes HTML first, then **bold**, bullet and numbered lists. */
function md(text) {
  const lines = esc(text).split("\n");
  let html = "", list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    let m;
    if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
      if (list !== "ul") { close(); html += "<ul>"; list = "ul"; }
      html += `<li>${m[1]}</li>`;
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
      if (list !== "ol") { close(); html += "<ol>"; list = "ol"; }
      html += `<li>${m[1]}</li>`;
    } else {
      close();
      if (line.trim()) html += `<p class="para">${line}</p>`;
    }
  }
  close();
  return html;
}

function nodeLabel(d) {
  const p = d.props || {};
  if (d.label === "Sensor") return p.type;
  return p.name || d.id;
}

/* ------------------------------------------------------------------ header + status */
function tickClock() {
  $("clock").textContent = new Date().toLocaleString([], {
    weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

async function loadHealth() {
  try {
    const h = await (await fetch("/health")).json();
    const item = (ok, label, value) => `<span><i class="dot ${ok ? "" : "fallback"}"></i>${label} <b>${esc(value)}</b></span>`;
    $("statusbar").innerHTML =
      item(h.graph_backend === "neo4j", "Graph store", h.graph_backend === "neo4j" ? "Neo4j" : "Local snapshot") +
      item(h.llm_ready, "Language model", h.llm_ready ? h.llm : "Offline – answers quote source documents") +
      item(h.embedder !== "hashing-384", "Embeddings", h.embedder) +
      item(true, "Failure model", "Random forest");
  } catch (e) {
    $("statusbar").innerHTML = `<span><i class="dot fallback"></i>Server unreachable</span>`;
  }
}

function updateMachineKpi() {
  const machines = graph.nodes.filter((n) => n.label === "Machine");
  $("kpiMachines").textContent = machines.length;
  const counts = {};
  machines.forEach((m) => { counts[m.props.status] = (counts[m.props.status] || 0) + 1; });
  const parts = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([s, c]) => `${c} ${s.toLowerCase()}`);
  $("kpiMachinesSub").textContent = parts.join(" · ");
}

/* ------------------------------------------------------------------ D3 graph */
function buildLegend() {
  $("legend").innerHTML = Object.keys(COLORS).map((l) =>
    `<button type="button" data-label="${l}" title="Show or hide ${l} nodes"><span class="swatch" style="background:${COLORS[l]}"></span>${l}</button>`
  ).join("");
  $("legend").querySelectorAll("button").forEach((el) => {
    el.addEventListener("click", () => {
      const l = el.dataset.label;
      hiddenLabels.has(l) ? hiddenLabels.delete(l) : hiddenLabels.add(l);
      el.classList.toggle("off", hiddenLabels.has(l));
      renderGraph();
    });
  });
}

async function loadGraph() {
  graph = await (await fetch("/graph-data")).json();
  updateMachineKpi();
  renderGraph();
}

function renderGraph() {
  const container = $("graph");
  container.innerHTML = "";
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 500;

  const nodes = graph.nodes.filter((n) => !hiddenLabels.has(n.label)).map((n) => ({ ...n }));
  const ids = new Set(nodes.map((n) => n.id));
  const links = graph.links.filter((l) => ids.has(l.source.id ?? l.source) && ids.has(l.target.id ?? l.target))
    .map((l) => ({ ...l, source: l.source.id ?? l.source, target: l.target.id ?? l.target }));

  svg = d3.select(container).append("svg").attr("viewBox", [0, 0, width, height]);
  const layer = svg.append("g");
  zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", (ev) => layer.attr("transform", ev.transform));
  svg.call(zoom);

  linkSel = layer.append("g").selectAll("line").data(links).join("line").attr("class", "link");
  linkSel.append("title").text((d) => d.type.replaceAll("_", " ").toLowerCase());

  nodeSel = layer.append("g").selectAll("g").data(nodes, (d) => d.id).join("g")
    .attr("class", (d) => `node ${d.label.toLowerCase()}`)
    .call(d3.drag()
      .on("start", (ev, d) => { if (!ev.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on("end", (ev, d) => { if (!ev.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on("click", (ev, d) => { ev.stopPropagation(); showNodeInfo(d.id); });

  nodeSel.filter((d) => d.label === "Machine").append("circle").attr("class", "ring").attr("r", RADIUS.Machine);
  nodeSel.append("circle").attr("class", "core").attr("r", (d) => RADIUS[d.label] || 7).attr("fill", (d) => COLORS[d.label] || "#999");
  nodeSel.append("title").text((d) => `${d.label} · ${nodeLabel(d)}`);
  nodeSel.filter((d) => ["Machine", "Technician", "Part"].includes(d.label))
    .append("text").attr("dx", (d) => (RADIUS[d.label] || 7) + 5).attr("dy", 4).text(nodeLabel);

  svg.on("click", () => $("nodeInfo").classList.add("hidden"));

  // Scale distances/repulsion to the panel so small screens don't push everything to the edges.
  const k = Math.max(0.45, Math.min(1, Math.sqrt((width * height) / (1000 * 650))));
  const LINK_DIST = { HAS_PART: 58, HAS_SENSOR: 46, REPORTS_FAILURE_ON: 38, APPLIES_TO: 130, MAINTAINED_BY: 100 };
  simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance((d) => (LINK_DIST[d.type] || 60) * k)
      .strength((d) => (d.type === "APPLIES_TO" ? 0.04 : 0.7)))
    .force("charge", d3.forceManyBody().strength((d) => (d.label === "Machine" ? -650 : -150) * k))  // many-body repulsion
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(width / 2).strength(0.045))
    .force("y", d3.forceY(height / 2).strength(0.065))
    .force("collide", d3.forceCollide().radius((d) => (RADIUS[d.label] || 7) + 4))
    .on("tick", () => {
      // Boundary constraint: invert velocity on SVG edge collision so nodes stay in the viewport.
      nodes.forEach((d) => {
        const r = RADIUS[d.label] || 7;
        if (d.x < r || d.x > width - r) { d.vx *= -1; d.x = Math.max(r, Math.min(width - r, d.x)); }
        if (d.y < r || d.y > height - r) { d.vy *= -1; d.y = Math.max(r, Math.min(height - r, d.y)); }
      });
      linkSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

  applyAlerts();
  highlight(lastHighlight);
}

function highlight(ids) {
  lastHighlight = ids || [];
  if (!nodeSel) return;
  const set = new Set(lastHighlight);
  const active = set.size > 0;
  nodeSel.classed("hl", (d) => set.has(d.id)).classed("dim", (d) => active && !set.has(d.id));
  linkSel.classed("hl", (d) => set.has(d.source.id ?? d.source) && set.has(d.target.id ?? d.target));
}

function applyAlerts() {
  if (nodeSel) nodeSel.classed("alert-pulse", (d) => alerting.has(d.id));
}

async function showNodeInfo(id) {
  const box = $("nodeInfo");
  try {
    const n = await (await fetch(`/node/${encodeURIComponent(id)}`)).json();
    const title = n.name || n.title || n.type || id;
    const rows = Object.entries(n).filter(([k]) => !["label", "embedder"].includes(k))
      .map(([k, v]) => `<dt>${esc(k.replaceAll("_", " "))}</dt><dd>${esc(Array.isArray(v) ? v.join(", ") : v)}</dd>`).join("");
    box.innerHTML = `<div class="ni-type">${esc(n.label)}</div><div class="ni-title">${esc(title)}</div><dl>${rows}</dl>`;
    box.classList.remove("hidden");
  } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------------ query panel */
function renderSuggestions() {
  $("suggestions").innerHTML = SUGGESTIONS.map(([q, kind]) =>
    `<li data-q="${esc(q)}"><span>${esc(q)}</span><span class="kind">${kind}</span></li>`).join("");
  $("suggestions").querySelectorAll("li").forEach((li) => li.addEventListener("click", () => runQuery(li.dataset.q)));
}

function resultCard(query) {
  $("emptyState")?.remove();
  const card = document.createElement("article");
  card.className = "result pending";
  card.innerHTML = `<div class="result-head"><span class="result-q">${esc(query)}</span><span class="result-time">${timeNow()}</span></div>
    <div class="result-body">Running query…</div>`;
  $("results").prepend(card);
  $("results").scrollTop = 0;
  return card;
}

async function runQuery(query) {
  const card = resultCard(query);
  $("runBtn").disabled = true;
  $("progress").classList.add("active");
  try {
    const res = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const r = await res.json();
    sessionId = r.session_id;

    let details = `<div class="detail-row">Engine: ${esc(r.mode)}</div>`;
    if (r.entities && r.entities.carried_over) details += `<div class="detail-row">Subject taken from the previous query.</div>`;
    if (r.cypher) details += `<div class="detail-row">Cypher</div><pre>${esc(r.cypher)}</pre>`;
    if (r.sources && r.sources.length) {
      details += `<div class="detail-row">Retrieved documents</div><pre>${r.sources
        .map((s) => `${esc(s.id)}   similarity ${Number(s.score).toFixed(2)}   ${esc(s.source)}`).join("\n")}</pre>`;
    }
    card.className = "result";
    card.querySelector(".result-body").innerHTML = md(r.answer);
    card.insertAdjacentHTML("beforeend", `<div class="result-foot">
      <span class="source ${esc(r.route)}">Source: ${SOURCE_LABEL[r.route] || esc(r.route)}</span>
      <details><summary>Query details</summary>${details}</details></div>`);
    highlight(r.highlight);
  } catch (e) {
    card.className = "result";
    card.querySelector(".result-body").textContent = `The query could not be completed (${e.message}).`;
  } finally {
    $("runBtn").disabled = false;
    $("progress").classList.remove("active");
  }
}

/* ------------------------------------------------------------------ at-risk */
async function loadRisk() {
  const parts = await (await fetch("/predict-maintenance")).json();
  $("kpiRisk").textContent = parts.length;
  $("kpiRisk").classList.toggle("warn", parts.length > 0);
  $("riskList").innerHTML = parts.length ? parts.map((p) => `
    <li data-ids="${esc([p.part_id, p.machine_id, ...p.tickets].join(","))}" title="Highlight in the asset graph">
      <div><span class="part">${esc(p.part)}</span><small>${esc(p.machine)} · last failure ${esc((p.last_failure || "").slice(0, 10))}</small></div>
      <span class="count">${p.ticket_count} tickets</span>
    </li>`).join("") : "<li>No components currently at risk.</li>";
  $("riskList").querySelectorAll("li[data-ids]").forEach((li) =>
    li.addEventListener("click", () => highlight(li.dataset.ids.split(","))));
}

/* ------------------------------------------------------------------ telemetry */
function connectTelemetry() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`);
  ws.onopen = () => { $("wsState").textContent = "Streaming · 3 s interval"; };
  ws.onclose = () => { $("wsState").textContent = "Reconnecting…"; setTimeout(connectTelemetry, 3000); };
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    alerting.clear();
    let top = null;
    $("telemetryBody").innerHTML = data.machines.map((m) => {
      if (m.alert) alerting.add(m.machine_id);
      if (!top || m.failure_probability > top.failure_probability) top = m;
      const p = m.failure_probability;
      const cls = p >= 0.6 ? "high" : p >= 0.3 ? "mid" : "";
      return `<tr class="${m.alert ? "alert" : ""}">
        <td><span class="mid">${esc(m.machine_id)}</span>${esc(m.name)}</td>
        <td>${m.alert ? '<span class="pill alert">Over temperature</span>' : '<span class="pill ok">Normal</span>'}</td>
        <td class="num temp">${m.temperature_c.toFixed(1)}</td>
        <td class="num">${m.vibration_mm_s.toFixed(2)}</td>
        <td class="num">${m.rpm.toLocaleString()}</td>
        <td><div class="risk"><span class="track"><span class="fill ${cls}" style="width:${Math.max(2, p * 100)}%"></span></span>${(p * 100).toFixed(0)}%</div></td>
      </tr>`;
    }).join("");

    $("kpiAlerts").textContent = alerting.size;
    $("kpiAlerts").classList.toggle("danger", alerting.size > 0);
    $("kpiAlertsSub").textContent = alerting.size ? [...alerting].join(", ") + " above 90 °C" : "Temperature above 90 °C";
    if (top) {
      const p = top.failure_probability;
      $("kpiMl").textContent = `${(p * 100).toFixed(0)}%`;
      $("kpiMl").className = `kpi-value ${p >= 0.6 ? "danger" : p >= 0.3 ? "warn" : ""}`;
      $("kpiMlSub").textContent = `${top.name} (${top.machine_id})`;
    }
    // Temperatures above 90 °C mark the machine node in the graph.
    applyAlerts();
  };
}

/* ------------------------------------------------------------------ boot */
$("queryForm").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = $("queryInput").value.trim();
  if (q) { $("queryInput").value = ""; runQuery(q); }
});
$("resetView").addEventListener("click", () => {
  highlight([]);
  $("nodeInfo").classList.add("hidden");
  svg && svg.transition().duration(400).call(zoom.transform, d3.zoomIdentity);
});
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderGraph, 250); });

tickClock();
setInterval(tickClock, 30000);
buildLegend();
renderSuggestions();
loadHealth();
setInterval(loadHealth, 30000);
loadGraph().catch((e) => { $("graph").innerHTML = `<p style="padding:16px">The asset graph could not be loaded (${esc(e.message)}).</p>`; });
loadRisk();
connectTelemetry();
