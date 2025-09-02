from __future__ import annotations
import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, f1_score
import matplotlib.pyplot as plt
import mlflow
from joblib import dump

def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def anomaly_scores(model, X):
    # Higher score => more anomalous
    if hasattr(model, "score_samples"):
        s = -model.score_samples(X)
    else:
        # decision_function: positive for inliers, negative for outliers
        s = -getattr(model, "decision_function")(X)
    return s

def evaluate_scores(y_true, scores, k=None, threshold=None):
    metrics = {}
    if y_true is not None:
        # y_true: 1 = outlier
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, scores)
        except Exception:
            pass
        if k is not None:
            idx = np.argsort(scores)[::-1][:k]
            y_pred_at_k = np.zeros_like(scores, dtype=int)
            y_pred_at_k[idx] = 1
            metrics["precision_at_k"] = precision_score(y_true, y_pred_at_k, zero_division=0)
        if threshold is not None:
            y_pred = (scores >= threshold).astype(int)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    return metrics

def save_figure(fig, out_dir, name):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def save_model(model, out_dir, name):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"{name}.joblib")
    dump(model, path)
    return path

def log_run_to_mlflow(params: dict, metrics: dict, artifacts: list[str], run_name: str | None = None):
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        for art in artifacts:
            if os.path.exists(art):
                mlflow.log_artifact(art)
