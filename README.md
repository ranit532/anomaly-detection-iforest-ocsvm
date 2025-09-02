
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

## 🧠 Understanding Anomaly Detection

Anomaly detection is the process of finding data points that are rare and different from the rest of the data. Think of it like finding a needle in a haystack. These "needles" are the anomalies, and they can represent interesting events like credit card fraud, a malfunctioning sensor, or a rare disease.

This project uses two popular algorithms for anomaly detection: **Isolation Forest** and **One-Class SVM**.

### Isolation Forest: The "Easy to Isolate" Principle

The main idea behind Isolation Forest is that **anomalies are few and different, which makes them easy to isolate**.

Imagine you are in a crowded room and you need to find a specific person. If that person is wearing a bright yellow hat (an anomaly), you can easily find them with a single question: "Who is wearing the yellow hat?". However, if the person looks like everyone else, you would need to ask many more questions to identify them.

The Isolation Forest algorithm works in a similar way. It builds a "forest" of many decision trees. In each tree, it randomly selects a feature (e.g., age, purchase amount) and a random value to split the data. This process is repeated until every data point is isolated.

-   **Anomalies** are like the person with the yellow hat. They are easy to isolate and will have a short path from the root of the tree to the isolated point.
-   **Normal points** are like the person who looks like everyone else. They are harder to isolate and will have a much longer path in the tree.

The algorithm calculates the average path length for each data point across all the trees in the forest. Points with a short average path length are flagged as anomalies.

### One-Class SVM: Building a "Bubble" Around Normal Data

One-Class SVM works by trying to separate the data from the origin. It learns a boundary that encloses the majority of the data points, which are considered "normal". Any data point that falls outside this boundary is considered an anomaly.

Think of it as drawing a "bubble" around the normal data points. Everything inside the bubble is normal, and everything outside is an anomaly. The algorithm can even learn complex, non-spherical shapes for the bubble using a technique called the "kernel trick".

### When to use which?

-   **Isolation Forest**: It's fast, works well with large datasets and many features, and doesn't require much configuration. It's a great first choice for many problems.
-   **One-Class SVM**: It's very flexible and can learn complex boundaries around the normal data. It's a good choice for smaller datasets where the boundary between normal and anomalous data might be complex.

---

## 🧪 Real-World Use Cases

Anomaly detection is used in many industries to find critical events:

-   **Fraud Detection**: Identify fraudulent credit card transactions by flagging unusual spending patterns, such as abnormally large purchases or transactions in unusual locations.
-   **Network Intrusion Detection**: Detect cyber attacks by identifying suspicious activity in network traffic, like a sudden surge in data transfer or connections from unusual IP addresses.
-   **Predictive Maintenance**: Predict when a machine is likely to fail by detecting anomalies in sensor data from the equipment. This allows for maintenance to be scheduled before the machine breaks down.
-   **Healthcare**: Detect medical problems early by identifying anomalies in a patient's vital signs or lab results.
-   **Manufacturing**: Ensure product quality by detecting defects in products during the manufacturing process.

> ⚠️ Always validate anomalies with domain knowledge and human review. False positives can be costly.

---

## 🖼️ Example Output

Here is an example of the output generated by the script. The plot shows the decision boundary of the Isolation Forest model and highlights the detected anomalies in yellow.

![Isolation Forest Anomaly Detection](images/iforest_blobs_20250902-180629.png)

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
python -m scripts.train --model iforest --dataset blobs --contamination 0.05 --seed 42
# One-Class SVM on moons (nu=0.05, rbf)
python -m scripts.train --model ocsvm --dataset moons --nu 0.05 --gamma scale --seed 42
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
