const queryInput = document.getElementById("query");
const topKInput = document.getElementById("topK");
const seedInput = document.getElementById("seed");
const maxHopsInput = document.getElementById("maxHops");
const beamWidthInput = document.getElementById("beamWidth");
const pruneThresholdInput = document.getElementById("pruneThreshold");
const importanceWeightInput = document.getElementById("importanceWeight");
const recencyWeightInput = document.getElementById("recencyWeight");
const graphWeightInput = document.getElementById("graphWeight");
const embeddingWeightInput = document.getElementById("embeddingWeight");
const lexicalWeightInput = document.getElementById("lexicalWeight");
const rerankTopNInput = document.getElementById("rerankTopN");
const hybridEnabledInput = document.getElementById("hybridEnabled");
const rerankEnabledInput = document.getElementById("rerankEnabled");
const graphLimitInput = document.getElementById("graphLimit");
const graphLinkDistanceInput = document.getElementById("graphLinkDistance");
const graphRepulsionInput = document.getElementById("graphRepulsion");
const graphNodeScaleInput = document.getElementById("graphNodeScale");

const topKVal = document.getElementById("topKVal");
const maxHopsVal = document.getElementById("maxHopsVal");
const beamWidthVal = document.getElementById("beamWidthVal");
const pruneThresholdVal = document.getElementById("pruneThresholdVal");
const importanceWeightVal = document.getElementById("importanceWeightVal");
const recencyWeightVal = document.getElementById("recencyWeightVal");
const graphWeightVal = document.getElementById("graphWeightVal");
const embeddingWeightVal = document.getElementById("embeddingWeightVal");
const lexicalWeightVal = document.getElementById("lexicalWeightVal");
const rerankTopNVal = document.getElementById("rerankTopNVal");
const graphLimitVal = document.getElementById("graphLimitVal");
const graphLinkDistanceVal = document.getElementById("graphLinkDistanceVal");
const graphRepulsionVal = document.getElementById("graphRepulsionVal");
const graphNodeScaleVal = document.getElementById("graphNodeScaleVal");

const turnList = document.getElementById("turnList");
const appliedParamsEl = document.getElementById("appliedParams");
const statusBannerEl = document.getElementById("statusBanner");

const runQueryBtn = document.getElementById("runQuery");
const loadGraphBtn = document.getElementById("loadGraph");
const ingestSampleBtn = document.getElementById("ingestSample");
const ingestHistoryBtn = document.getElementById("ingestHistory");

let lastGraphData = null;
let busyCounter = 0;

runQueryBtn.addEventListener("click", runQuery);
loadGraphBtn.addEventListener("click", loadGraph);
ingestSampleBtn.addEventListener("click", ingestSample);
ingestHistoryBtn.addEventListener("click", ingestHistory);

bindSlider(topKInput, topKVal, (v) => `${Number(v)}`);
bindSlider(maxHopsInput, maxHopsVal, (v) => `${Number(v)}`);
bindSlider(beamWidthInput, beamWidthVal, (v) => `${Number(v)}`);
bindSlider(pruneThresholdInput, pruneThresholdVal, (v) => Number(v).toFixed(2));
bindSlider(importanceWeightInput, importanceWeightVal, (v) => Number(v).toFixed(2));
bindSlider(recencyWeightInput, recencyWeightVal, (v) => Number(v).toFixed(2));
bindSlider(graphWeightInput, graphWeightVal, (v) => Number(v).toFixed(2));
bindSlider(embeddingWeightInput, embeddingWeightVal, (v) => Number(v).toFixed(2));
bindSlider(lexicalWeightInput, lexicalWeightVal, (v) => Number(v).toFixed(2));
bindSlider(rerankTopNInput, rerankTopNVal, (v) => `${Number(v)}`);
bindSlider(graphLimitInput, graphLimitVal, (v) => `${Number(v)}`);
bindSlider(graphLinkDistanceInput, graphLinkDistanceVal, (v) => `${Number(v)}`, rerenderGraphIfLoaded);
bindSlider(graphRepulsionInput, graphRepulsionVal, (v) => `${Number(v)}`, rerenderGraphIfLoaded);
bindSlider(graphNodeScaleInput, graphNodeScaleVal, (v) => Number(v).toFixed(1), rerenderGraphIfLoaded);
setStatus("준비 완료. 1) Ingest Agent History 2) Run Query 3) Load Graph 순서로 진행하세요.", "info");

async function ingestSample() {
  if (isBusy()) return;
  setStatus("Sample ingest 실행 중...", "info");
  pushBusy();
  const payload = { path: "/workspace/data/samples/conversation.jsonl" };
  try {
    const res = await requestJson("/ingest/jsonl", payload, { timeoutMs: 120000 });
    if (!res.ok) return;
    setStatus("Sample ingest 완료. Query를 확인하고 Load Graph로 그래프를 보세요.", "success");
    await runQuery();
    await loadGraph();
  } finally {
    popBusy();
  }
}

async function ingestHistory() {
  if (isBusy()) return;
  setStatus("History ingest job 생성 중...", "info");
  pushBusy();
  const payload = {
    source: "both",
    max_files: 500,
  };
  try {
    const startRes = await requestJson("/ingest/history/start", payload, { timeoutMs: 30000 });
    if (!startRes.ok) return;
    const job = startRes.data || {};
    const jobId = job.job_id;
    if (!jobId) {
      setStatus("History ingest job_id를 받지 못했습니다.", "error");
      return;
    }
    await pollIngestJob(jobId);
  } finally {
    popBusy();
  }
}

async function pollIngestJob(jobId) {
  let consecutiveFailures = 0;
  while (true) {
    const res = await requestJson(`/ingest/jobs/${encodeURIComponent(jobId)}`, null, {
      method: "GET",
      timeoutMs: 30000,
    });
    if (!res.ok) {
      consecutiveFailures += 1;
      if (res.status === 404) {
        setStatus(
          `History ingest 상태를 찾을 수 없습니다(job_id=${jobId}). 서버가 재시작되었을 수 있어요. 다시 Ingest Agent History를 눌러주세요.`,
          "error"
        );
        return;
      }
      if (consecutiveFailures >= 10) {
        setStatus(
          `History ingest 상태 확인이 계속 실패합니다(job_id=${jobId}). 네트워크/서버 상태를 확인한 뒤 다시 시도하세요.`,
          "error"
        );
        return;
      }
      await sleep(Math.min(5000, 1200 * consecutiveFailures));
      continue;
    }
    consecutiveFailures = 0;
    const job = res.data || {};
    const status = String(job.status || "");
    const progress = job.progress || {};

    const processed = Number(progress.turns_processed || 0);
    const total = progress.turns_total === null || progress.turns_total === undefined ? null : Number(progress.turns_total);
    const pct = total ? ` (${Math.floor((processed / Math.max(1, total)) * 100)}%)` : "";

    if (status === "succeeded") {
      setStatus(
        `History ingest 완료: turns=${processed}, entities=${Number(progress.extracted_entities || 0)}, relations=${Number(progress.extracted_relations || 0)}. 다음: 1) Query 입력 2) Run Query 3) Graph Seed 입력 후 Load Graph`,
        "success"
      );
      return;
    }
    if (status === "failed") {
      setStatus(`History ingest 실패: ${job.error || "unknown error"}`, "error");
      return;
    }
    if (status === "cancelled") {
      setStatus("History ingest 취소됨.", "error");
      return;
    }

    const phase = String(progress.phase || status || "running");
    setStatus(
      `History ingest ${phase}... turns ${processed}/${total ?? "?"}${pct} | entities ${Number(
        progress.extracted_entities || 0
      )} | relations ${Number(progress.extracted_relations || 0)}`,
      "info"
    );
    await sleep(2000);
  }
}

async function runQuery() {
  if (isBusy()) return;
  const queryText = queryInput.value.trim();
  if (!queryText) {
    setStatus("Query가 비어있습니다. 입력 후 Run Query를 눌러주세요.", "error");
    return;
  }
  const rerankHint = rerankEnabledInput.checked
    ? " (리랭커 최초 로딩은 모델 다운로드로 몇 분 걸릴 수 있어요)"
    : "";
  setStatus(`Query 실행 중...${rerankHint}`, "info");
  pushBusy();
  const payload = {
    query: queryText,
    top_k: Number(topKInput.value),
    max_hops: Number(maxHopsInput.value),
    beam_width: Number(beamWidthInput.value),
    prune_threshold: Number(pruneThresholdInput.value),
    hybrid_enabled: hybridEnabledInput.checked,
    graph_weight: Number(graphWeightInput.value),
    embedding_weight: Number(embeddingWeightInput.value),
    lexical_weight: Number(lexicalWeightInput.value),
    importance_weight: Number(importanceWeightInput.value),
    recency_weight: Number(recencyWeightInput.value),
    recall_half_life_hours: 72,
    rerank_enabled: rerankEnabledInput.checked,
    rerank_top_n: Number(rerankTopNInput.value),
  };
  try {
    const res = await requestJson("/query", payload, {
      timeoutMs: rerankEnabledInput.checked ? 15 * 60 * 1000 : 2 * 60 * 1000,
    });
    if (!res.ok) {
      turnList.innerHTML = `<li class="turn-item">Query failed: ${escapeHtml(res.text || "")}</li>`;
      appliedParamsEl.textContent = "";
      return;
    }
    const data = res.data || {};
    renderAppliedParams(data.applied_params || payload);
    renderTurns(data.selected_turns || []);
    setStatus(`Query 완료: ${data.selected_turns?.length || 0}개 턴을 찾았습니다.`, "info");
  } finally {
    popBusy();
  }
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
    `hybrid=${Boolean(params.hybrid_enabled)}`,
    `w(g/e/l)=${formatNumber(params.graph_weight)}/${formatNumber(params.embedding_weight)}/${formatNumber(params.lexical_weight)}`,
    `importance_w=${formatNumber(params.importance_weight)}`,
    `recency_w=${formatNumber(params.recency_weight)}`,
    `rerank=${Boolean(params.rerank_enabled)} top_n=${Number(params.rerank_top_n || 0)}`,
    `reranker_ready=${Boolean(params.reranker_available)}`,
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
      `fusion ${formatNumber(breakdown.fusion_score)}`,
      `graph ${formatNumber(breakdown.graph_signal)}`,
      `emb ${formatNumber(breakdown.embedding_signal)}`,
      `lex ${formatNumber(breakdown.lexical_signal)}`,
      `rerank ${formatNumber(breakdown.rerank_component)}`,
      `+ imp ${formatNumber(breakdown.importance_component)}`,
      `+ rec ${formatNumber(breakdown.recency_component)}`,
    ].join(" ");
    const recalled = turn.last_recalled_at ? `last recall ${formatDateTime(turn.last_recalled_at)}` : "last recall -";
    const chunkMeta = `chunks ${Number(turn.chunk_count || 1)} (${turn.chunk_profile || "default"})`;

    li.innerHTML = `
      <div class="turn-meta">${escapeHtml(turn.conversation_id)} / ${escapeHtml(turn.turn_id)} / score ${Number(turn.score || 0).toFixed(3)}</div>
      <div class="turn-text">${escapeHtml(turn.text)}</div>
      <div class="turn-stats">importance ${formatNumber(turn.importance_score)} | recency ${formatNumber(turn.recency_factor)} | ${escapeHtml(recalled)}</div>
      <div class="turn-stats">${escapeHtml(chunkMeta)}</div>
      <div class="turn-stats">${escapeHtml(breakdownText)}</div>
      <div class="turn-reason">${escapeHtml(reasons || "direct entity match")}</div>
    `;
    turnList.appendChild(li);
  }
}

async function loadGraph() {
  if (isBusy()) return;
  const seed = seedInput.value.trim();
  if (!seed) {
    setStatus("Graph Seed가 비어있습니다. 입력 후 Load Graph를 눌러주세요.", "error");
    return;
  }

  setStatus("Graph 로딩 중...", "info");
  pushBusy();
  const graphLimit = Number(graphLimitInput.value);
  try {
    const res = await requestJson(`/graph/subgraph?seed=${encodeURIComponent(seed)}&limit=${graphLimit}`, null, {
      method: "GET",
      timeoutMs: 120000,
    });
    if (!res.ok) return;
    const graph = res.data || {};
    lastGraphData = graph;
    const d3Ready = await ensureD3Loaded();
    if (!d3Ready) {
      renderGraphPlaceholder(
        `그래프 렌더러(d3) 로딩 실패: 외부 CDN 접근이 막혀 있을 수 있어요.\n` +
          `Top Turns 리스트는 정상 사용 가능하고, 그래프는 지금은 비활성화됩니다.`
      );
      return;
    }
    renderGraph(graph.nodes || [], graph.edges || []);
    setStatus(`Graph 로드 완료: nodes=${graph.nodes?.length || 0}, edges=${graph.edges?.length || 0}`, "info");
  } finally {
    popBusy();
  }
}

function renderGraph(nodes, edges) {
  const container = document.getElementById("graphCanvas");
  container.innerHTML = "";

  const width = container.clientWidth || 700;
  const height = 470;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("class", "graph-svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const panSurface = svg
    .append("rect")
    .attr("class", "pan-surface")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", width)
    .attr("height", height);

  const viewport = svg.append("g").attr("class", "graph-viewport");

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(edges)
        .id((d) => d.id)
        .distance(Number(graphLinkDistanceInput.value))
    )
    .force("charge", d3.forceManyBody().strength(-Number(graphRepulsionInput.value)))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(20));

  const link = viewport
    .append("g")
    .selectAll("line")
    .data(edges)
    .join("line")
    .attr("class", "link");

  const node = viewport
    .append("g")
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", "node")
    .attr("r", (d) => (8 + Math.max(0, d.score || 0) * 4) * Number(graphNodeScaleInput.value))
    .attr("fill", (d) => (d.kind === "entity" ? "#0f766e" : "#7c3aed"))
    .call(
      d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
    );

  const labels = viewport
    .append("g")
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .text((d) => d.label || d.id);

  const edgeLabels = viewport
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

  const zoom = d3
    .zoom()
    .scaleExtent([0.2, 4])
    .filter((event) => {
      const targetTag = event?.target?.tagName?.toLowerCase?.() || "";
      if (event.type === "wheel") return true;
      if (event.type === "mousedown") return targetTag !== "circle" && event.button === 0;
      return targetTag !== "circle";
    })
    .on("start", (event) => {
      if (event.sourceEvent?.type === "mousedown") svg.classed("is-panning", true);
    })
    .on("zoom", (event) => {
      viewport.attr("transform", event.transform);
    })
    .on("end", () => {
      svg.classed("is-panning", false);
    });

  svg.call(zoom);
  svg.on("dblclick.zoom", null);
  panSurface.on("dblclick", null);

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

function renderGraphPlaceholder(message) {
  const container = document.getElementById("graphCanvas");
  container.innerHTML = "";
  const box = document.createElement("div");
  box.style.padding = "0.8rem";
  box.style.color = "#475569";
  box.style.fontSize = "0.9rem";
  box.style.whiteSpace = "pre-wrap";
  box.textContent = message;
  container.appendChild(box);
}

function bindSlider(input, output, formatter, onChange = null) {
  const refresh = () => {
    output.textContent = formatter(input.value);
    if (onChange) onChange();
  };
  input.addEventListener("input", refresh);
  refresh();
}

function rerenderGraphIfLoaded() {
  if (!lastGraphData) return;
  renderGraph(lastGraphData.nodes || [], lastGraphData.edges || []);
}

function setStatus(message, level = "info") {
  statusBannerEl.textContent = message;
  statusBannerEl.className = `status-banner status-${level}`;
}

function setBusy(isBusyFlag) {
  const buttons = [runQueryBtn, loadGraphBtn, ingestSampleBtn, ingestHistoryBtn];
  for (const button of buttons) {
    button.disabled = Boolean(isBusyFlag);
  }
}

function isBusy() {
  return busyCounter > 0;
}

function pushBusy() {
  busyCounter += 1;
  setBusy(true);
}

function popBusy() {
  busyCounter = Math.max(0, busyCounter - 1);
  setBusy(busyCounter > 0);
}

function describeError(err) {
  if (!err) return "unknown error";
  if (err.name === "AbortError") return "request timed out";
  if (typeof err === "string") return err;
  return err.message || String(err);
}

async function requestJson(url, payload, options = {}) {
  const {
    method = payload === null ? "GET" : "POST",
    timeoutMs = 120000,
    headers = payload === null ? {} : { "Content-Type": "application/json" },
  } = options;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(url, {
      method,
      headers,
      body: payload === null ? undefined : JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_) {
      data = null;
    }

    if (!res.ok) {
      setStatus(`${method} ${url} 실패: ${text || res.status}`, "error");
    }

    return { ok: res.ok, status: res.status, text, data };
  } catch (err) {
    setStatus(`네트워크 오류: ${describeError(err)}`, "error");
    return { ok: false, status: 0, text: "", data: null };
  } finally {
    clearTimeout(timer);
  }
}

async function ensureD3Loaded() {
  if (window.d3) return true;

  const candidates = [
    "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
    "https://unpkg.com/d3@7/dist/d3.min.js",
  ];

  for (const src of candidates) {
    try {
      setStatus("그래프 렌더러(d3) 로딩 중...", "info");
      await loadScript(src, 15000);
      if (window.d3) return true;
    } catch (_) {
      // try next
    }
  }

  setStatus("그래프 렌더러(d3) 로딩 실패. 외부 네트워크 접근을 확인하세요.", "error");
  return false;
}

function loadScript(src, timeoutMs) {
  return new Promise((resolve, reject) => {
    const existing = Array.from(document.getElementsByTagName("script")).find((s) => s.src === src);
    if (existing) {
      existing.addEventListener("load", () => resolve(true), { once: true });
      existing.addEventListener("error", () => reject(new Error("script load failed")), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    const timer = setTimeout(() => {
      script.remove();
      reject(new Error("script load timeout"));
    }, timeoutMs);

    script.addEventListener(
      "load",
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      { once: true }
    );
    script.addEventListener(
      "error",
      () => {
        clearTimeout(timer);
        reject(new Error("script load failed"));
      },
      { once: true }
    );

    document.head.appendChild(script);
  });
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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
