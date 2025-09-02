from __future__ import annotations
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_model(name: str, **kwargs):
    name = name.lower()
    if name in {"iforest", "isolationforest", "isolation_forest"}:
        model = IsolationForest(**kwargs)
        return model
    elif name in {"ocsvm", "oneclasssvm", "one_class_svm"}:
        oc = OneClassSVM(**kwargs)
        pipe = Pipeline([("scaler", StandardScaler()), ("ocsvm", oc)])
        return pipe
    else:
        raise ValueError(f"Unknown model: {name}")
