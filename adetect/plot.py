from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _to_2d(X):
    if X.shape[1] == 2:
        return X
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X)

def decision_contour_2d(model, X, grid_steps=300, padding=0.2):
    X2 = _to_2d(X)
    x_min, x_max = X2[:,0].min(), X2[:,0].max()
    y_min, y_max = X2[:,1].min(), X2[:,1].max()
    dx = (x_max - x_min) * padding
    dy = (y_max - y_min) * padding
    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, grid_steps),
        np.linspace(y_min - dy, y_max + dy, grid_steps),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    try:
        # Prefer score_samples if available (higher = more normal), else decision_function
        if hasattr(model, "score_samples"):
            Z = model.score_samples(grid)
        else:
            Z = model.decision_function(grid)
    except Exception:
        Z = None

    fig = plt.figure()
    ax = plt.gca()
    if Z is not None:
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=20, alpha=0.6)  # default colormap
    ax.scatter(X2[:,0], X2[:,1], s=10)
    ax.set_xlabel("Feature 1 (proj)")
    ax.set_ylabel("Feature 2 (proj)")
    ax.set_title("Decision Surface & Data (2D projection)")
    return fig, ax
