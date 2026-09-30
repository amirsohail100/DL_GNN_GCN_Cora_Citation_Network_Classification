"""
We are going to create our API now. 
"""
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import onnxruntime as ort


CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, 'simple_gcn_cora.onnx')
DATA_DIR    = os.path.join(BASE_DIR, "data", "Planetoid")
STATIC_DIR  = os.path.join(BASE_DIR, "static")

model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI()


class GraphPredictRequest(BaseModel):
    node_features: List[List[float]]
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int]


def softmax(scores: np.ndarray):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return):
    output = model_session.run(
        ["logits"],
        {
            'node_features': node_features.astype(np.float32),
            'edge_indices': edge_index.astype(np.int64)
        },
    )
    logits = output[0]
    probabilites = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            "node_index": i,
            "predicted_class_id": int(predicted_classes[i]),
            "predicted_class_name": CORA_CLASSES[int(predicted_classes[i])],
            "probabilites": probabilites[i].tolist(),
            "logits": logits[i].tolist(),
        })

    return {
        'num_nodes': node_features.shape[0],
        "num_edges": edge_index.shape[1],
        "predictions": results
    }


@app.get('/')
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Simple GCN Cora API", "status": "Running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "providers": model_session.get_providers()}


@app.get('/info')
def model_info():
    return {
        "Model Name": "SimpeGCN",
        "feature_dimension": 1433,
        "num_classes": 7,
        "class_mapping": CORA_CLASSES,
        "inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs": [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    if not request.node_features or request.node_features == 0:
        raise HTTPException(400, "Node Features cannot be empty")

    for feature_vector in request.node_features:
        if len(feature_vector) != 1433:
            raise HTTPException(422, "Each node's feature vector must exactly have 1433 features")

    node_features = np.array(request.node_features, dtype=np.float32)
    num_nodes = len(request.node_features)

    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge_indices must have shape [2, num_edges]")
    else:
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    all_node_features = list(range(num_nodes))
    return run_model(node_features, edge_index, all_node_features)


@app.post('/predict/cora_node')
def predict_real_cora_nodes(request: CoraNodeRequest):
    from torch_geometric.datasets import Planetoid
    try:
        cora_dataset = Planetoid(root=DATA_DIR, name="Cora")[0]
    except Exception as error:
        raise HTTPException(500, f"failed to load cora dataset: {error}")

    largest_valid_index = cora_dataset.num_nodes - 1
    invalid_index = [
        i for i in request.node_indices
        if i < 0 or i > largest_valid_index
    ]
    if invalid_index:
        raise HTTPException(
            400, f"node_index out of bound (must be 0 to {largest_valid_index})"
        )

    return run_model(
        cora_dataset.x.numpy(),
        cora_dataset.edge_index.numpy(),
        request.node_indices
    )


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")