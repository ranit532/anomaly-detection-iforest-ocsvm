from __future__ import annotations
import argparse, os
import numpy as np
from adetect import datasets as D, models as M, plot as P, utils as U

def parse_args():
    ap = argparse.ArgumentParser(description="Train anomaly detection model with MLflow logging.")
    ap.add_argument("--model", required=True, choices=["iforest", "ocsvm"], help="Model type")
    ap.add_argument("--dataset", required=True, choices=["blobs", "moons", "circles", "csv"], help="Dataset type")
    ap.add_argument("--data-path", default=None, help="Path to CSV when dataset=csv")
    ap.add_argument("--label-col", default=None, help="Label column name in CSV (1=outlier, 0=inlier)")
    ap.add_argument("--n-inliers", type=int, default=800)
    ap.add_argument("--n-outliers", type=int, default=40)
    ap.add_argument("--n-features", type=int, default=2)
    ap.add_argument("--contamination", type=float, default=0.05, help="Used for IF thresholding/logging")
    ap.add_argument("--nu", type=float, default=0.05, help="OCSVM nu")
    ap.add_argument("--kernel", default="rbf", help="OCSVM kernel")
    ap.add_argument("--gamma", default="scale", help="OCSVM gamma")
    ap.add_argument("--n-estimators", type=int, default=200, help="IF trees")
    ap.add_argument("--max-samples", default="auto", help="IF max_samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--models-dir", default="models")
    return ap.parse_args()

def get_dataset(args):
    if args.dataset == "blobs":
        X, y = D.make_blobs_with_outliers(n_inliers=args.n_inliers, n_outliers=args.n_outliers,
                                          n_features=args.n_features, random_state=args.seed)
    elif args.dataset == "moons":
        X, y = D.make_moons_with_outliers(n_inliers=args.n_inliers, n_outliers=args.n_outliers,
                                          n_features=args.n_features, random_state=args.seed)
    elif args.dataset == "circles":
        X, y = D.make_circles_with_outliers(n_inliers=args.n_inliers, n_outliers=args.n_outliers,
                                            n_features=args.n_features, random_state=args.seed)
    elif args.dataset == "csv":
        assert args.data_path, "--data-path is required when dataset=csv"
        X, y = D.load_from_csv(args.data_path, label_col=args.label_col)
    else:
        raise ValueError("Unknown dataset")
    return X, y

def build(args):
    if args.model == "iforest":
        model = M.build_model("iforest",
                              n_estimators=args.n_estimators,
                              max_samples=args.max_samples,
                              contamination=args.contamination,
                              random_state=args.seed)
    else:
        model = M.build_model("ocsvm", nu=args.nu, kernel=args.kernel, gamma=args.gamma)
    return model

def main():
    args = parse_args()
    X, y = get_dataset(args)
    model = build(args)
    model.fit(X)

    # Scores and thresholding
    scores = U.anomaly_scores(model, X)
    k_expected = int(round(args.contamination * len(X))) if y is None else int(np.sum(y == 1))
    if k_expected <= 0:
        k_expected = max(1, int(0.01 * len(X)))
    thresh = np.partition(scores, -k_expected)[-k_expected]

    # Evaluate (if labels available)
    metrics = U.evaluate_scores(y, scores, k=k_expected if y is not None else None, threshold=thresh)
    params = vars(args).copy()
    params.pop("images_dir", None); params.pop("models_dir", None)

    # Plotting (2D projection if needed)
    fig, ax = P.decision_contour_2d(model, X)
    # Mark predicted anomalies
    pred_idx = scores >= thresh
    X2 = X[:, :2] if X.shape[1] >= 2 else np.c_[X, np.zeros_like(X)]
    ax.scatter(X2[pred_idx,0], X2[pred_idx,1], s=20, marker="x")  # default color/style

    name = f"{args.model}_{args.dataset}_{U.timestamp()}"
    fig_path = U.save_figure(fig, args.images_dir, name)
    model_path = U.save_model(model, args.models_dir, name)

    # MLflow logging
    artifacts = [fig_path, model_path]
    U.log_run_to_mlflow(params=params, metrics=metrics, artifacts=artifacts, run_name=name)

    print("Saved figure:", fig_path)
    print("Saved model:", model_path)
    if metrics:
        print("Metrics:", metrics)

if __name__ == "__main__":
    main()
