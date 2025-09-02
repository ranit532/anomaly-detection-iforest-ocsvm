from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles

def _inject_outliers(X: np.ndarray, n_outliers: int, low=-10.0, high=10.0, rng=None):
    rng = np.random.default_rng(rng)
    n_features = X.shape[1]
    outliers = rng.uniform(low, high, size=(n_outliers, n_features))
    return outliers

def make_blobs_with_outliers(n_inliers=500, n_outliers=25, centers=3, cluster_std=1.2, n_features=2, random_state=42):
    X_in, _ = make_blobs(n_samples=n_inliers, centers=centers, cluster_std=cluster_std,
                         n_features=n_features, random_state=random_state)
    X_out = _inject_outliers(X_in, n_outliers=n_outliers, rng=random_state)
    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(len(X_in), dtype=int), np.ones(len(X_out), dtype=int)])  # 1 = outlier
    return X, y

def make_moons_with_outliers(n_inliers=500, n_outliers=25, noise=0.05, n_features=2, random_state=42):
    X_in, _ = make_moons(n_samples=n_inliers, noise=noise, random_state=random_state)
    if n_features > 2:
        # pad extra dims with small noise
        extra = np.random.default_rng(random_state).normal(scale=0.1, size=(n_inliers, n_features - 2))
        X_in = np.hstack([X_in, extra])
    X_out = _inject_outliers(X_in, n_outliers=n_outliers, rng=random_state)
    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(len(X_in), dtype=int), np.ones(len(X_out), dtype=int)])
    return X, y

def make_circles_with_outliers(n_inliers=500, n_outliers=25, noise=0.05, factor=0.5, n_features=2, random_state=42):
    X_in, _ = make_circles(n_samples=n_inliers, noise=noise, factor=factor, random_state=random_state)
    if n_features > 2:
        extra = np.random.default_rng(random_state).normal(scale=0.1, size=(n_inliers, n_features - 2))
        X_in = np.hstack([X_in, extra])
    X_out = _inject_outliers(X_in, n_outliers=n_outliers, rng=random_state)
    X = np.vstack([X_in, X_out])
    y = np.hstack([np.zeros(len(X_in), dtype=int), np.ones(len(X_out), dtype=int)])
    return X, y

def load_from_csv(path: str, label_col: str | None = None):
    df = pd.read_csv(path)
    if label_col is not None and label_col in df.columns:
        y = df[label_col].values
        X = df.drop(columns=[label_col]).values
    else:
        y = None
        X = df.values
    return X, y
