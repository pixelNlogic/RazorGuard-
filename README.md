# 🛡️ RazorGuard — AI-Powered Fraud Risk Manager

**A machine learning application for detecting potentially fraudulent credit card transactions and evaluating fraud-detection performance.**

RazorGuard is a deployed fraud-detection dashboard built with Python and Streamlit. It uses Logistic Regression with feature scaling to identify potentially fraudulent transactions from an anonymized credit card transaction dataset.

The project focuses on more than just predictions: it also evaluates precision, recall, false positives, false negatives, and the trade-offs involved in fraud detection.

**🚀 Live Demo:** [Open RazorGuard](https://razorguard.streamlit.app/)

---

## 📌 Project Overview

Credit card fraud is a challenging classification problem because fraudulent transactions represent only a tiny fraction of all transactions. A useful detection system must identify fraud while limiting unnecessary alerts on legitimate transactions.

RazorGuard demonstrates an end-to-end machine learning workflow, from model training and test-set evaluation to an interactive web dashboard.

### Key Features

- **Fraud Prediction:** Classifies transactions using a trained machine learning model.
- **Interactive Dashboard:** Provides a web interface built with Streamlit.
- **Model Performance Metrics:** Displays precision, recall, F1 score, and average precision.
- **Confusion Matrix Analysis:** Shows true positives, true negatives, false positives, and false negatives.
- **Adjustable Decision Threshold:** Allows users to explore how the fraud-classification threshold affects predictions.
- **Cost Analysis:** Demonstrates the potential financial impact of false positives and false negatives using configurable illustrative cost assumptions.
- **Model Persistence:** Saves the trained model and evaluation metrics for reuse.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset handling and preprocessing |
| NumPy | Numerical computation |
| Scikit-learn | Model training and evaluation |
| StandardScaler | Feature scaling |
| Logistic Regression | Fraud classification |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web dashboard |
| Git & GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

---

## 📊 Model Performance

RazorGuard was evaluated on a held-out test set containing **56,962 transactions**.

| Metric | Test Result |
|---|---:|
| Precision | 39.01% |
| Recall | 88.78% |
| F1 Score | 54.21% |
| Average Precision | 71.90% |

### Confusion Matrix Results

| Outcome | Transactions |
|---|---:|
| True Negatives | 56,728 |
| False Positives | 136 |
| False Negatives | 11 |
| True Positives | 87 |

### What These Metrics Mean

- **Precision:** Approximately 39.01% of transactions flagged as fraudulent in the test set were actually fraudulent.
- **Recall:** The model detected approximately 88.78% of fraudulent transactions in the test set.
- **F1 Score:** Balances precision and recall.
- **Average Precision:** Summarizes precision-recall performance across classification thresholds.

The model prioritizes detecting a high proportion of fraudulent transactions, while false positives remain an important consideration.

*These results are from a held-out test set, not a guarantee of future performance. The dataset is anonymized and publicly available; production deployment would require additional validation, monitoring, security controls, and evaluation on representative real-world data.*

---

## 🏗️ Project Structure

```text
RazorGuard/
├── app.py                     # Streamlit dashboard
├── train_model.py             # Model training and evaluation
├── requirements.txt           # Python dependencies
├── models/
│   ├── fraud_model.joblib     # Trained model
│   └── metrics.json           # Test-set metrics
├── data/
│   └── raw/
│       └── creditcard.csv     # Local dataset (not tracked by Git)
├── .gitignore
└── README.md
```

*The raw dataset is not included in the Git repository because of its size. The local data file is expected at `data/raw/creditcard.csv` when training the model.*

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/pixelNlogic/RazorGuard-.git
cd RazorGuard-
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The local application will normally be available at `http://localhost:8501/`.

### Optional: Train the model again

Place the dataset at `data/raw/creditcard.csv`, then run:

```bash
python train_model.py
```

This regenerates the trained model and evaluation metrics according to the training script.

---

## 🧪 Dataset

This project uses the publicly available **Credit Card Fraud Detection** dataset, which contains anonymized transaction features and a binary target label indicating legitimate or fraudulent transactions.

The dataset is highly imbalanced, making precision-recall evaluation especially relevant.

**Dataset source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🔍 What I Learned

Through RazorGuard, I worked on:

- Building a machine learning classification workflow.
- Handling an imbalanced fraud-detection dataset.
- Evaluating models using precision, recall, F1 score, and average precision.
- Understanding false-positive and false-negative trade-offs.
- Persisting trained models and metrics.
- Developing an interactive dashboard with Streamlit.
- Deploying a Python application and managing code with Git and GitHub.

---

## 🚧 Future Improvements

- Compare Logistic Regression with additional classification models.
- Add precision-recall curves and threshold comparison visualizations.
- Explore imbalance-handling techniques and probability calibration.
- Add automated tests and model-validation checks.
- Improve monitoring, explainability, and model robustness.
- Evaluate on additional representative datasets before any real-world use.

---

## ⚠️ Disclaimer

RazorGuard is an educational machine learning prototype. It is not a production-ready payment security system and should not be used to make real financial decisions without further validation, security review, and operational safeguards.

The cost analysis uses illustrative assumptions; it does not represent verified savings or actual merchant losses prevented.

---

## 👩‍💻 Author

**Varshvi Singh**  
Computer Science Engineering Student

Interested in software development, machine learning, and building practical technology projects.

- GitHub: [@pixelNlogic](https://github.com/pixelNlogic)
- Project Repository: [RazorGuard](https://github.com/pixelNlogic/RazorGuard-)

---

⭐ If you find this project interesting, consider starring the repository!
