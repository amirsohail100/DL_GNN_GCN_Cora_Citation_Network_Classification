"""
Cora GCN Explorer — Streamlit UI
Talks to the FastAPI service (main.py) that serves the ONNX GCN model.

Run with:
    streamlit run streamlit_app.py
"""

import json
import random

import requests
import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------------
# Config / constants (mirrors the FastAPI service)
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Cora GCN Explorer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

CLASS_COLORS = {
    0: "#e3a548",  # Case_Based
    1: "#7fb77e",  # Genetic_Algorithms
    2: "#4fd1c5",  # Neural_Networks
    3: "#8c7ae6",  # Probabilistic_Methods
    4: "#e0705f",  # Reinforcement_Learning
    5: "#5b9bd5",  # Rule_Learning
    6: "#d488b9",  # Theory
}

FEATURE_DIM = 1433

# ----------------------------------------------------------------------------
# Theming — dark, matching the HTML build (near-black bg, amber + teal accents)
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,500;0,600;1,500&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  { font-family: 'JetBrains Mono', monospace; }

    .stApp {
        background:
          radial-gradient(circle at 12% 8%, rgba(227,165,72,0.06), transparent 40%),
          radial-gradient(circle at 88% 92%, rgba(79,209,197,0.06), transparent 40%),
          #0a0c10;
        color: #e8e6df;
    }

    h1, h2, h3 { font-family: 'Newsreader', serif !important; font-weight: 600 !important; color: #f4f2ea !important; }

    .eyebrow {
        display:inline-flex; align-items:center; gap:8px; font-size:11px; letter-spacing:0.14em;
        text-transform:uppercase; color:#4fd1c5; border:1px solid rgba(79,209,197,0.3);
        background:rgba(79,209,197,0.06); padding:5px 12px; border-radius:100px; margin-bottom:14px;
    }

    section[data-testid="stSidebar"] {
        background: #12151c; border-right: 1px solid #232733;
    }

    div[data-testid="stMetric"] {
        background: #161a22; border: 1px solid #232733; border-radius: 12px; padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] { color: #8b90a0 !important; }

    .card {
        background:#161a22; border:1px solid #232733; border-radius:12px;
        padding:18px 20px; margin-bottom:14px;
    }
    .node-tag { font-size:11px; color:#565c6d; letter-spacing:0.05em; }
    .predicted-name { font-family:'Newsreader', serif; font-size:19px; font-weight:600; }
    .conf-text { font-size:11px; color:#8b90a0; }

    .legend-chip {
        display:inline-flex; align-items:center; gap:6px; font-size:11px; color:#8b90a0;
        border:1px solid #1b1f29; padding:5px 10px; border-radius:100px; background:#161a22; margin:2px;
    }
    .legend-sw { width:8px; height:8px; border-radius:50%; display:inline-block; }

    .stButton > button {
        background:#e3a548; color:#171207; border:none; font-weight:700; border-radius:9px;
        font-family:'JetBrains Mono', monospace; letter-spacing:0.02em;
    }
    .stButton > button:hover { filter:brightness(1.08); color:#171207; }

    div[data-baseweb="tab-list"] { gap: 4px; }
    button[data-baseweb="tab"] { font-family:'JetBrains Mono', monospace; }

    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background:#161a22 !important; border:1px solid #232733 !important; color:#e8e6df !important;
        font-family:'JetBrains Mono', monospace !important;
    }

    hr { border-color: #1b1f29 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — API connection
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Connection")
    api_base = st.text_input("API base URL", value="https://cora-project.onrender.com").rstrip("/")

    if st.button("Check health", use_container_width=True):
        st.session_state["_check_health"] = True

    health_placeholder = st.empty()

    def render_health():
        try:
            res = requests.get(f"{api_base}/health", timeout=5)
            res.raise_for_status()
            data = res.json()
            providers = ", ".join(data.get("providers", [])) or "ready"
            health_placeholder.success(f"online · {providers}")
        except Exception as exc:  # noqa: BLE001
            health_placeholder.error(f"unreachable: {exc}")

    render_health()

    st.markdown("---")
    st.markdown("### Model")
    st.caption("SimpleGCN · ONNX Runtime")
    st.markdown(
        f"""
        - **Feature dimension:** {FEATURE_DIM}
        - **Classes:** {len(CORA_CLASSES)}
        - **Dataset:** Cora citation network
        """
    )
    st.markdown("---")
    st.markdown("### Topic legend")
    legend_html = "".join(
        f'<span class="legend-chip"><span class="legend-sw" style="background:{CLASS_COLORS[i]}"></span>{name.replace("_"," ")}</span>'
        for i, name in CORA_CLASSES.items()
    )
    st.markdown(legend_html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.markdown('<div class="eyebrow">◆ graph neural network · node classification</div>', unsafe_allow_html=True)
st.title("Cora GCN Explorer")
st.caption(
    "Classify citation-network papers into one of seven research topics using a trained "
    "graph convolutional network, served over ONNX Runtime."
)
st.write("")

# ----------------------------------------------------------------------------
# Result rendering helpers
# ----------------------------------------------------------------------------

def render_predictions(data: dict):
    col1, col2 = st.columns(2)
    col1.metric("Nodes in graph", data.get("num_nodes", "–"))
    col2.metric("Edges in graph", data.get("num_edges", "–"))
    st.write("")

    for pred in data.get("predictions", []):
        class_id = pred["predicted_class_id"]
        color = CLASS_COLORS[class_id]
        name = pred["predicted_class_name"].replace("_", " ")
        conf = pred["probabilities"][class_id] * 100

        with st.container():
            st.markdown(
                f"""
                <div class="card">
                    <div class="node-tag">NODE #{pred['node_index']}</div>
                    <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                        <span style="width:10px;height:10px;border-radius:3px;background:{color};display:inline-block;"></span>
                        <span class="predicted-name">{name}</span>
                        <span class="conf-text">{conf:.1f}% confidence</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            df = pd.DataFrame(
                {
                    "topic": [c.replace("_", " ") for c in CORA_CLASSES.values()],
                    "probability": pred["probabilities"],
                    "class_id": list(CORA_CLASSES.keys()),
                }
            ).sort_values("probability", ascending=True)

            for _, row in df.iterrows():
                bar_color = CLASS_COLORS[int(row["class_id"])]
                pct = row["probability"] * 100
                bcol1, bcol2, bcol3 = st.columns([2.2, 5, 1])
                with bcol1:
                    st.markdown(
                        f"<span style='font-size:12px;color:{'#f4f2ea' if int(row['class_id'])==class_id else '#8b90a0'}'>{row['topic']}</span>",
                        unsafe_allow_html=True,
                    )
                with bcol2:
                    st.markdown(
                        f"""
                        <div style="background:#1b1f29;border-radius:100px;height:8px;overflow:hidden;margin-top:4px;">
                            <div style="width:{pct:.1f}%;height:100%;background:{bar_color};border-radius:100px;"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with bcol3:
                    st.markdown(f"<span style='font-size:11px;color:#565c6d;'>{pct:.1f}%</span>", unsafe_allow_html=True)

            with st.expander("Raw logits"):
                st.code(json.dumps([round(v, 4) for v in pred["logits"]]), language="json")

            st.write("")


# ----------------------------------------------------------------------------
# Main tabs — real Cora nodes vs. custom graph
# ----------------------------------------------------------------------------

tab_cora, tab_custom = st.tabs(["Real Cora papers", "Custom graph"])

with tab_cora:
    st.markdown("Query real papers from the Cora citation dataset by index (0–2707).")
    indices_raw = st.text_input("Node indices (comma-separated)", value="0, 1, 2, 3", key="cora_indices")

    if st.button("Classify these papers", key="run_cora"):
        try:
            indices = [int(x.strip()) for x in indices_raw.split(",") if x.strip() != ""]
            if not indices:
                raise ValueError("enter at least one node index")
        except ValueError as exc:
            st.error(f"Invalid node indices: {exc}")
        else:
            with st.spinner("Running inference over the Cora graph…"):
                try:
                    res = requests.post(
                        f"{api_base}/predict/cora_node",
                        json={"node_indices": indices},
                        timeout=60,
                    )
                    res.raise_for_status()
                    render_predictions(res.json())
                except requests.exceptions.RequestException as exc:
                    detail = ""
                    try:
                        detail = res.json().get("detail", "")  # type: ignore[possibly-undefined]
                    except Exception:  # noqa: BLE001
                        pass
                    st.error(f"Request failed: {detail or exc}")

with tab_custom:
    st.markdown(f"Submit your own node features (each a vector of **{FEATURE_DIM}** numbers) and optional edges.")

    gen_col1, gen_col2 = st.columns([1, 3])
    with gen_col1:
        gen_count = st.number_input("Random nodes", min_value=1, max_value=20, value=4, step=1)
    with gen_col2:
        st.write("")
        st.write("")
        generate = st.button("Generate random graph", key="gen_random")

    if generate:
        feats = [
            [1 if random.random() > 0.94 else 0 for _ in range(FEATURE_DIM)]
            for _ in range(gen_count)
        ]
        src, dst = [], []
        for i in range(gen_count):
            j = (i + 1) % gen_count
            src += [i, j]
            dst += [j, i]
        st.session_state["custom_features"] = json.dumps(feats)
        st.session_state["custom_edges"] = json.dumps([src, dst])

    features_raw = st.text_area(
        "Node features (JSON array of arrays)",
        value=st.session_state.get("custom_features", ""),
        height=140,
        key="custom_features",
        placeholder='[[0, 0, 1, ...], [0, 1, 0, ...]]',
    )
    edges_raw = st.text_area(
        "Edge indices — optional, shape [2, num_edges]",
        value=st.session_state.get("custom_edges", ""),
        height=90,
        key="custom_edges",
        placeholder="[[0, 1, 2], [1, 2, 0]]  (leave blank for self-loops only)",
    )
    st.caption("If edges are omitted, the API self-loops every node — no message passing between nodes.")

    if st.button("Classify custom graph", key="run_custom"):
        try:
            features = json.loads(features_raw)
            if not isinstance(features, list) or not features:
                raise ValueError("must be a non-empty array of arrays")
            bad = [i for i, v in enumerate(features) if len(v) != FEATURE_DIM]
            if bad:
                raise ValueError(f"nodes at index {bad} do not have {FEATURE_DIM} features")
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid node features: {exc}")
        else:
            payload = {"node_features": features}
            if edges_raw.strip():
                try:
                    edges = json.loads(edges_raw)
                    payload["edge_indices"] = edges
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid edge indices JSON: {exc}")
                    payload = None

            if payload:
                with st.spinner("Running inference…"):
                    try:
                        res = requests.post(f"{api_base}/predict", json=payload, timeout=60)
                        res.raise_for_status()
                        render_predictions(res.json())
                    except requests.exceptions.RequestException as exc:
                        detail = ""
                        try:
                            detail = res.json().get("detail", "")  # type: ignore[possibly-undefined]
                        except Exception:  # noqa: BLE001
                            pass
                        st.error(f"Request failed: {detail or exc}")

st.markdown("---")
st.caption(f"Cora GCN Explorer · talks to `{api_base}` · 7-class topic model over {FEATURE_DIM}-dim bag-of-words features")
