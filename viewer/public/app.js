const queryInput = document.getElementById("query");
const topKInput = document.getElementById("topK");
const seedInput = document.getElementById("seed");
const maxHopsInput = document.getElementById("maxHops");
const beamWidthInput = document.getElementById("beamWidth");
const pruneThresholdInput = document.getElementById("pruneThreshold");
const importanceWeightInput = document.getElementById("importanceWeight");
const recencyWeightInput = document.getElementById("recencyWeight");

const topKVal = document.getElementById("topKVal");
const maxHopsVal = document.getElementById("maxHopsVal");
const beamWidthVal = document.getElementById("beamWidthVal");
const pruneThresholdVal = document.getElementById("pruneThresholdVal");
const importanceWeightVal = document.getElementById("importanceWeightVal");
const recencyWeightVal = document.getElementById("recencyWeightVal");

const turnList = document.getElementById("turnList");
const appliedParamsEl = document.getElementById("appliedParams");

const runQueryBtn = document.getElementById("runQuery");
const loadGraphBtn = document.getElementById("loadGraph");
const ingestSampleBtn = document.getElementById("ingestSample");

runQueryBtn.addEventListener("click", runQuery);
loadGraphBtn.addEventListener("click", loadGraph);
ingestSampleBtn.addEventListener("click", ingestSample);

bindSlider(topKInput, topKVal, (v) => `${Number(v)}`);
bindSlider(maxHopsInput, maxHopsVal, (v) => `${Number(v)}`);
bindSlider(beamWidthInput, beamWidthVal, (v) => `${Number(v)}`);
bindSlider(pruneThresholdInput, pruneThresholdVal, (v) => Number(v).toFixed(2));
bindSlider(importanceWeightInput, importanceWeightVal, (v) => Number(v).toFixed(2));
bindSlider(recencyWeightInput, recencyWeightVal, (v) => Number(v).toFixed(2));

async function ingestSample() {
  const payload = { path: "/workspace/data/samples/conversation.jsonl" };
  const res = await fetch("/ingest/jsonl", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    alert(`ingest failed: ${await res.text()}`);
    return;
  }
  await runQuery();
  await loadGraph();
}

async function runQuery() {
  const payload = {
    query: queryInput.value,
    top_k: Number(topKInput.value),
    max_hops: Number(maxHopsInput.value),
    beam_width: Number(beamWidthInput.value),
    prune_threshold: Number(pruneThresholdInput.value),
    importance_weight: Number(importanceWeightInput.value),
    recency_weight: Number(recencyWeightInput.value),
    recall_half_life_hours: 72,
  };
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    turnList.innerHTML = `<li class="turn-item">Query failed: ${await res.text()}</li>`;
    appliedParamsEl.textContent = "";
    return;
  }

  const data = await res.json();
  renderAppliedParams(data.applied_params || payload);
  renderTurns(data.selected_turns || []);
}

function renderAppliedParams(params) {
  if (!params || Object.keys(params).length === 0) {
    appliedParamsEl.textContent = "";
    return;
  }
  const text = [
    `hops=${params.max_hops}`,
    `beam=${params.beam_width}`,
    `prune=${formatNumber(params.prune_threshold)}`,
    `importance_w=${formatNumber(params.importance_weight)}`,
    `recency_w=${formatNumber(params.recency_weight)}`,
    `top_k=${params.top_k}`,
  ].join(" | ");
  appliedParamsEl.textContent = text;
}

function renderTurns(turns) {
  turnList.innerHTML = "";
  if (!turns.length) {
    turnList.innerHTML = '<li class="turn-item">No turns found.</li>';
    return;
  }

  for (const turn of turns) {
    const li = document.createElement("li");
    li.className = "turn-item";

    const reasons = (turn.path_summary || [])
      .slice(0, 3)
      .map((step) => `${step.from_entity_name} -[${step.relation_type}]-> ${step.to_entity_name}`)
      .join(" | ");

    const breakdown = turn.score_breakdown || {};
    const breakdownText = [
      `base ${formatNumber(breakdown.base_score)}`,
      `+ imp ${formatNumber(breakdown.importance_component)}`,
      `+ rec ${formatNumber(breakdown.recency_component)}`,
    ].join(" ");
    const recalled = turn.last_recalled_at ? `last recall ${formatDateTime(turn.last_recalled_at)}` : "last recall -";

    li.innerHTML = `
      <div class="turn-meta">${escapeHtml(turn.conversation_id)} / ${escapeHtml(turn.turn_id)} / score ${Number(turn.score || 0).toFixed(3)}</div>
      <div class="turn-text">${escapeHtml(turn.text)}</div>
      <div class="turn-stats">importance ${formatNumber(turn.importance_score)} | recency ${formatNumber(turn.recency_factor)} | ${escapeHtml(recalled)}</div>
      <div class="turn-stats">${escapeHtml(breakdownText)}</div>
      <div class="turn-reason">${escapeHtml(reasons || "direct entity match")}</div>
    `;
    turnList.appendChild(li);
  }
}

async function loadGraph() {
  const seed = seedInput.value.trim();
  if (!seed) return;

  const res = await fetch(`/graph/subgraph?seed=${encodeURIComponent(seed)}&limit=150`);
  if (!res.ok) {
    alert(`graph load failed: ${await res.text()}`);
    return;
  }

  const graph = await res.json();
  renderGraph(graph.nodes || [], graph.edges || []);
}

function renderGraph(nodes, edges) {
  const container = document.getElementById("graphCanvas");
  container.innerHTML = "";

  const width = container.clientWidth || 700;
  const height = 470;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(edges)
        .id((d) => d.id)
        .distance(95)
    )
    .force("charge", d3.forceManyBody().strength(-220))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(20));

  const link = svg
    .append("g")
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("class", "link");

  const node = svg
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "node")
    .attr("r", (d) => 8 + Math.max(0, d.score || 0) * 4)
    .attr("fill", (d) => (d.kind === "entity" ? "#0f766e" : "#7c3aed"))
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  const labels = svg
    .append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .text((d) => d.label || d.id);

  const edgeLabels = svg
    .append("g")
    .selectAll("text")
    .data(edges)
    .join("text")
    .attr("class", "node-label")
    .attr("font-size", 9)
    .attr("fill", "#64748b")
    .text((d) => d.label || "");

  node.append("title").text((d) => `${d.label} (${d.id})`);
  link.append("title").text((d) => (d.evidence_turn_ids || []).join(", "));

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

    labels.attr("x", (d) => d.x + 10).attr("y", (d) => d.y + 3);

    edgeLabels
      .attr("x", (d) => (d.source.x + d.target.x) / 2)
      .attr("y", (d) => (d.source.y + d.target.y) / 2);
  });

  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
}

function bindSlider(input, output, formatter) {
  const refresh = () => {
    output.textContent = formatter(input.value);
  };
  input.addEventListener("input", refresh);
  refresh();
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatNumber(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0.000";
  return num.toFixed(3);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
