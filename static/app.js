// Topic Class Colors & Icons
const TOPIC_CONFIG = {
  "Neural_Networks": { color: "#6366f1", icon: "fa-brain" },
  "Probabilistic_Methods": { color: "#06b6d4", icon: "fa-chart-column" },
  "Reinforcement_Learning": { color: "#10b981", icon: "fa-robot" },
  "Theory": { color: "#f59e0b", icon: "fa-book-open" },
  "Genetic_Algorithms": { color: "#ec4899", icon: "fa-dna" },
  "Rule_Learning": { color: "#8b5cf6", icon: "fa-scale-balanced" },
  "Case_Based": { color: "#f43f5e", icon: "fa-briefcase" }
};

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initTabs();
  initCoraExplorer();
  initCustomPredictor();
  initModelInfo();
});

// --- Health Check ---
async function initHealthCheck() {
  const statusBadge = document.getElementById("status-badge");
  const statusText = document.getElementById("status-text");

  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "healthy") {
      statusText.textContent = "ONNX Model Online";
      statusBadge.style.borderColor = "rgba(16, 185, 129, 0.4)";
    } else {
      statusText.textContent = "Model Offline";
      statusBadge.querySelector(".status-dot").style.backgroundColor = "#f43f5e";
    }
  } catch (err) {
    statusText.textContent = "API Disconnected";
    statusBadge.querySelector(".status-dot").style.backgroundColor = "#f43f5e";
  }
}

// --- Tabs ---
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabId = btn.getAttribute("data-tab");
      
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(tabId).classList.add("active");
    });
  });
}

// --- Cora Explorer ---
function initCoraExplorer() {
  const nodeInput = document.getElementById("node-id-input");
  const btnPredict = document.getElementById("btn-predict-cora");
  const btnRandom = document.getElementById("btn-random-node");
  const chips = document.querySelectorAll(".chip[data-node]");

  btnPredict.addEventListener("click", () => {
    const nodeId = parseInt(nodeInput.value, 10);
    if (!isNaN(nodeId) && nodeId >= 0 && nodeId <= 2707) {
      predictCoraNode(nodeId);
    } else {
      alert("Please enter a valid node ID between 0 and 2707.");
    }
  });

  btnRandom.addEventListener("click", () => {
    const randomId = Math.floor(Math.random() * 2708);
    nodeInput.value = randomId;
    predictCoraNode(randomId);
  });

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const nodeId = parseInt(chip.getAttribute("data-node"), 10);
      nodeInput.value = nodeId;
      predictCoraNode(nodeId);
    });
  });

  // Initial predict for Node #0
  predictCoraNode(0);
}

async function predictCoraNode(nodeId) {
  const emptyState = document.getElementById("cora-empty-state");
  const resultsContainer = document.getElementById("cora-results-container");
  const predTopicTitle = document.getElementById("pred-topic-title");
  const predTopicName = document.getElementById("pred-topic-name");
  const predConfidencePct = document.getElementById("pred-confidence-pct");
  const probBarsList = document.getElementById("prob-bars-list");
  const rawLogitsCode = document.getElementById("raw-logits-code");
  const graphNodeCount = document.getElementById("graph-node-count");

  graphNodeCount.textContent = `Node #${nodeId} Citation Context`;

  try {
    const res = await fetch("/predict/cora_node", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_indices: [nodeId] })
    });

    if (!res.ok) throw new Error(`HTTP Error ${res.status}`);

    const data = await res.json();
    const pred = data.predictions[0];

    emptyState.classList.add("hidden");
    resultsContainer.classList.remove("hidden");

    const topicName = pred.predicted_class_name;
    const topicCfg = TOPIC_CONFIG[topicName] || { color: "#6366f1", icon: "fa-graduation-cap" };
    
    // Set Header Card
    predTopicName.textContent = topicName.replace(/_/g, " ");
    predTopicTitle.querySelector("i").className = `fa-solid ${topicCfg.icon}`;
    predTopicTitle.style.color = topicCfg.color;

    const probs = pred.probabilities || pred.probabilites || [];
    const maxProb = probs.length ? Math.max(...probs) : 0;
    predConfidencePct.textContent = `${(maxProb * 100).toFixed(1)}%`;

    // Render Probabilities
    probBarsList.innerHTML = "";
    
    // Pair probabilities with class mapping
    const classMapping = {
      0: "Case_Based", 1: "Genetic_Algorithms", 2: "Neural_Networks",
      3: "Probabilistic_Methods", 4: "Reinforcement_Learning", 5: "Rule_Learning", 6: "Theory"
    };

    const paired = probs.map((prob, idx) => ({
      name: classMapping[idx] || `Class_${idx}`,
      prob: prob,
      isTop: prob === maxProb
    }));

    // Sort by probability descending
    paired.sort((a, b) => b.prob - a.prob);

    paired.forEach(item => {
      const cfg = TOPIC_CONFIG[item.name] || { color: "#6366f1" };
      const pctStr = (item.prob * 100).toFixed(1);

      const probItem = document.createElement("div");
      probItem.className = "prob-item";
      probItem.innerHTML = `
        <div class="prob-info">
          <span class="prob-name">${item.name.replace(/_/g, " ")}</span>
          <span class="prob-pct">${pctStr}%</span>
        </div>
        <div class="prob-bar-track">
          <div class="prob-bar-fill ${item.isTop ? 'highest' : ''}" style="width: 0%; background: ${item.isTop ? cfg.color : ''}"></div>
        </div>
      `;
      probBarsList.appendChild(probItem);

      // Trigger CSS width transition
      setTimeout(() => {
        probItem.querySelector(".prob-bar-fill").style.width = `${pctStr}%`;
      }, 50);
    });

    rawLogitsCode.textContent = JSON.stringify(pred.logits, null, 2);

    // Draw Citation Graph Canvas
    drawCitationGraph(nodeId, topicCfg.color);

  } catch (err) {
    console.error("Prediction failed:", err);
    alert(`Failed to get GCN prediction for node ${nodeId}: ${err.message}`);
  }
}

// Draw Canvas Citation Network
function drawCitationGraph(centralNodeId, nodeColor) {
  const canvas = document.getElementById("citation-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;

  // Generate 6 simulated cited neighbor nodes
  const numNeighbors = 6;
  const neighbors = [];
  const radius = 80;

  for (let i = 0; i < numNeighbors; i++) {
    const angle = (i * 2 * Math.PI) / numNeighbors;
    neighbors.push({
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
      id: (centralNodeId + (i + 1) * 37) % 2708
    });
  }

  // Draw Edge Lines
  ctx.lineWidth = 1.5;
  neighbors.forEach(n => {
    ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(n.x, n.y);
    ctx.stroke();
  });

  // Draw Central Node
  ctx.fillStyle = nodeColor;
  ctx.shadowColor = nodeColor;
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(centerX, centerY, 18, 0, 2 * Math.PI);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Central Node Label
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(`#${centralNodeId}`, centerX, centerY);

  // Draw Neighbor Nodes
  neighbors.forEach(n => {
    ctx.fillStyle = "rgba(30, 41, 59, 0.9)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(`#${n.id}`, n.x, n.y);
  });
}

// --- Custom Predictor ---
function initCustomPredictor() {
  const numNodesSelect = document.getElementById("num-custom-nodes");
  const radioButtons = document.querySelectorAll("input[name='feature-mode']");
  const payloadEditor = document.getElementById("custom-payload-editor");
  const btnPredictCustom = document.getElementById("btn-predict-custom");

  function updatePayload() {
    const numNodes = parseInt(numNodesSelect.value, 10);
    const mode = document.querySelector("input[name='feature-mode']:checked").value;

    const nodeFeatures = [];
    for (let i = 0; i < numNodes; i++) {
      const vec = new Array(1433).fill(0.0);
      if (mode === "preset_ai") {
        // Activate words associated with Neural Nets / AI
        [10, 25, 42, 100, 250, 500, 800, 1200].forEach(idx => vec[idx] = 1.0);
      } else if (mode === "preset_theory") {
        // Activate words associated with Math / Theory
        [5, 15, 30, 90, 300, 600, 900, 1400].forEach(idx => vec[idx] = 1.0);
      } else {
        // Random sparse indicators
        for (let k = 0; k < 10; k++) {
          vec[Math.floor(Math.random() * 1433)] = 1.0;
        }
      }
      nodeFeatures.push(vec);
    }

    // Build edges (chain or triangle)
    let edges = [[0, 1], [1, 0]];
    if (numNodes === 3) {
      edges = [[0, 1, 2, 1], [1, 2, 0, 0]];
    } else if (numNodes === 5) {
      edges = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]];
    }

    const payload = {
      node_features: nodeFeatures,
      edge_indices: edges
    };

    payloadEditor.value = JSON.stringify(payload, null, 2);
  }

  numNodesSelect.addEventListener("change", updatePayload);
  radioButtons.forEach(r => r.addEventListener("change", updatePayload));
  updatePayload();

  btnPredictCustom.addEventListener("click", async () => {
    const resultsWrapper = document.getElementById("custom-results-wrapper");
    const resultsGrid = document.getElementById("custom-nodes-results-grid");

    try {
      const payload = JSON.parse(payloadEditor.value);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errJson = await res.json();
        throw new Error(errJson.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      resultsWrapper.classList.remove("hidden");
      resultsGrid.innerHTML = "";

      data.predictions.forEach(pred => {
        const topicName = pred.predicted_class_name;
        const cfg = TOPIC_CONFIG[topicName] || { color: "#6366f1", icon: "fa-file" };
        const probs = pred.probabilities || pred.probabilites || [];
        const maxProb = probs.length ? Math.max(...probs) : 0;

        const card = document.createElement("div");
        card.className = "card glass-card";
        card.innerHTML = `
          <div class="card-header">
            <h3><i class="fa-solid ${cfg.icon}" style="color: ${cfg.color}"></i> Paper #${pred.node_index + 1}</h3>
          </div>
          <div class="card-body">
            <div style="font-size: 1.1rem; font-weight: 700; color: ${cfg.color}; margin-bottom: 0.5rem">
              ${topicName.replace(/_/g, " ")}
            </div>
            <div style="font-size: 0.85rem; color: #9ca3af">
              Confidence: <strong style="color: #10b981">${(maxProb * 100).toFixed(1)}%</strong>
            </div>
          </div>
        `;
        resultsGrid.appendChild(card);
      });

    } catch (err) {
      alert(`Custom prediction error: ${err.message}`);
    }
  });
}

// --- Model Info & Benchmark ---
async function initModelInfo() {
  const specFile = document.getElementById("spec-file");
  const specProvider = document.getElementById("spec-provider");
  const specInputX = document.getElementById("spec-input-x");
  const specInputEdge = document.getElementById("spec-input-edge");
  const specOutput = document.getElementById("spec-output");
  const btnBenchmark = document.getElementById("btn-run-benchmark");
  const benchMs = document.getElementById("bench-ms");
  const benchDetails = document.getElementById("bench-details");

  try {
    const res = await fetch("/info");
    const data = await res.json();
    const modelName = data["Model Name"] || data.model_name || "SimpleGCN";
    specFile.textContent = `${modelName} (simple_gcn_cora.onnx)`;
    if (data.inputs && data.inputs.length >= 2) {
      specInputX.textContent = `${data.inputs[0].name}: [num_nodes, ${data.feature_dimension}]`;
      specInputEdge.textContent = `${data.inputs[1].name}: [2, num_edges]`;
    }
    if (data.outputs && data.outputs.length >= 1) {
      specOutput.textContent = `${data.outputs[0].name}: [num_nodes, ${data.num_classes}]`;
    }
  } catch (err) {
    console.error("Failed to load model info:", err);
  }

  btnBenchmark.addEventListener("click", async () => {
    btnBenchmark.disabled = true;
    btnBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Benchmark...';
    benchMs.textContent = "Calculating...";

    const timings = [];
    try {
      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        await fetch("/predict/cora_node", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ node_indices: [0, 10, 42, 100, 500] })
        });
        const elapsed = performance.now() - start;
        timings.push(elapsed);
      }

      const avgLatency = (timings.reduce((a, b) => a + b, 0) / timings.length).toFixed(2);
      benchMs.textContent = `${avgLatency} ms`;
      benchDetails.innerHTML = `
        <p><i class="fa-solid fa-circle-check" style="color: #10b981"></i> Average end-to-end API response time over 5 runs: <strong>${avgLatency} ms</strong>.</p>
      `;

    } catch (err) {
      benchMs.textContent = "Error";
      alert("Benchmark failed.");
    } finally {
      btnBenchmark.disabled = false;
      btnBenchmark.innerHTML = '<i class="fa-solid fa-stopwatch"></i> Run Latency Benchmark';
    }
  });
}
