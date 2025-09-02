
# Anomaly Detection: Isolation Forest & One-Class SVM (scikit-learn + MLflow)

Production-ready, reproducible examples of **Isolation Forest** and **One-Class SVM** for anomaly detection using:
- Python 3.10+
- scikit-learn
- NumPy / pandas
- Matplotlib (pyplot) for graphs
- **MLflow** for experiment tracking and artifact logging

Both models are trained on synthetic 2D/ND datasets (so you can run end‑to‑end without downloading data) and produce decision boundary plots plus anomaly highlights saved into the `images/` folder and logged to MLflow.

---

## 📁 Project Structure

```
anomaly-detection-iforest-ocsvm/
├─ adetect/
│  ├─ __init__.py
│  ├─ datasets.py          # Synthetic datasets with controllable contamination
│  ├─ models.py            # Model builders: IsolationForest, OneClassSVM
│  ├─ plot.py              # Matplotlib helpers for decision contours & scatter
│  └─ utils.py             # Metrics, saving, MLflow helpers
├─ scripts/
│  └─ train.py             # CLI to run experiments end-to-end
├─ images/                 # Saved figures (populated after running scripts)
│  └─ .gitkeep
├─ models/                 # Saved joblib models per run
│  └─ .gitkeep
├─ mlruns/                 # Local MLflow backend store (UI reads from here)
│  └─ .gitkeep
├─ .gitignore
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 🧠 Concepts & Math (Crash Course)

### Isolation Forest (IF)

- IF builds an ensemble of random **isolation trees** (iTrees) over subsamples of the data.
- Anomalies are **easier to isolate** and therefore have **shorter average path lengths** in iTrees.
- Let \(h(x)\) be the path length of point \(x\) in an iTree; the anomaly score is:
  
  \[ s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}} \]
  
  where \(n\) is the subsample size and
  
  \[ c(n) = 2\,H(n-1) - \frac{2(n-1)}{n}, \quad H(m) = \sum_{i=1}^{m} \frac{1}{i} \]
  
- Interpretation: **higher \(s(x,n)\)** → more anomalous. A threshold derived from the **contamination** rate yields binary predictions.
- **Scaling**: near-linear in data size via subsampling → great for large datasets.

### One‑Class SVM (OCSVM)

- Finds a function \(f(x) = \operatorname{sign}(g(x))\) that is **positive for the majority (inliers)** and negative for outliers, in a high‑dimensional feature space via the kernel trick.
- Primal form (using feature map \(\Phi\)) with hyperparameters \(\nu \in (0,1]\) and slack \(\xi_i\):
  
  \[ \min_{w,\rho,\xi} \frac{1}{2}\lVert w \rVert^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho \]
  subject to \( (w \cdot \Phi(x_i)) \ge \rho - \xi_i, \; \xi_i \ge 0. \)
  
- Decision function: \( g(x) = w \cdot \Phi(x) - \rho \). Commonly uses **RBF kernel** with `gamma` controlling smoothness.
- **\(\nu\)** bounds the fraction of outliers (upper bound) and support vectors (lower bound).
- **Scaling**: training complexity grows superlinearly (often \(\mathcal{O}(n^3)\)); better for small/medium datasets.

### When to use which?

- **IF**: scalable, robust for tabular data, handles high dimensions; good first choice for large datasets.
- **OCSVM**: flexible boundaries via kernels; can model complex inlier manifolds; prefer when data is modest in size and kernelized geometry helps.

---

## 🧪 Real‑World Use Cases

- **Fraud detection**: flag abnormal card transactions or wallet top‑ups.
- **Network intrusion**: detect traffic patterns deviating from baseline.
- **Manufacturing**: discover sensor readings indicating equipment faults.
- **Healthcare**: identify outlying vitals or lab measurements for early alerts.
- **IT Ops**: spot unusual service latencies or error spikes.

> ⚠️ Always validate anomalies with domain knowledge and human review. False positives can be costly.

---

## ✅ Getting Started (Step‑by‑Step)

### 1) Clone and set up environment

```bash
git clone https://github.com/<you>/anomaly-detection-iforest-ocsvm.git
cd anomaly-detection-iforest-ocsvm

# (Recommended) Create a virtual environment
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run an experiment with MLflow tracking

The following trains a model on a synthetic dataset, logs parameters/metrics/artifacts to **MLflow**, and saves figures under `images/`:

```bash
# Isolation Forest on 2D blobs (5% contamination)
python scripts/train.py --model iforest --dataset blobs --contamination 0.05 --seed 42

# One-Class SVM on moons (nu=0.05, rbf)
python scripts/train.py --model ocsvm --dataset moons --nu 0.05 --gamma scale --seed 42
```

**Outputs**

- Figures saved to `images/` as `model_dataset_<timestamp>.png`
- Models saved under `models/` as `model_dataset_<timestamp>.joblib`
- MLflow run with metrics and artifacts in `./mlruns`

### 3) Explore runs in MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
# Then open http://127.0.0.1:5000 in your browser
```

---

## 📊 Metrics & Evaluation

For synthetic datasets with known outliers, we compute:

- **ROC AUC** on anomaly scores
- **Precision@k** where \(k\) matches expected number of outliers
- **F1** using threshold derived from contamination/nu

Plots show decision contours and highlight predicted anomalies (2D projection via PCA if >2D).

---

## ⚙️ Reproducibility

- All random states are seeded (`--seed`).
- Dependencies are pinned in `requirements.txt`.
- Artifacts and params/metrics are tracked in MLflow.

---

## 🔢 Key Hyperparameters

- **Isolation Forest**: `n_estimators`, `max_samples`, `contamination`, `bootstrap`
- **OCSVM**: `nu`, `kernel`, `gamma`
- **Common**: feature scaling (OCSVM uses `StandardScaler`), PCA projection for plotting

---

## 🧾 License

This project is released under the MIT License (see `LICENSE`).

---

## 🙋 FAQ

**Q: Where are the images?**  
A: After you run `scripts/train.py`, figures are saved to `images/` and also logged to MLflow.

**Q: Can I use my own CSV?**  
A: Yes. Add a loader in `adetect/datasets.py` and pass `--dataset mycsv --data-path path/to.csv`. Ensure a purely numerical matrix (categoricals encoded) and optional `--label-col` if ground truth is available.

**Q: My OCSVM is slow or runs out of memory.**  
A: Try subsampling, reducing features, or switching to IF for large data.

---

## 📚 References

- Liu, Ting, and Zhou. *Isolation Forest.* ICDM 2008.
- Schölkopf et al. *Estimating the Support of a High-Dimensional Distribution.* Neural Computation 2001.
