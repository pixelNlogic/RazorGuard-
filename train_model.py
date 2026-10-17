
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)

DATA_PATH = "data/raw/creditcard.csv"
MODEL_PATH = "models/fraud_model.joblib"
METRICS_PATH = "models/metrics.json"

# These are illustrative assumptions, not measured real-world losses.
FALSE_POSITIVE_COST = 500   # assumed cost of flagging a legitimate transaction (₹)
FALSE_NEGATIVE_COST = 2000  # assumed cost of missing a fraudulent transaction (₹)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Check the file location."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' in the dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Total transactions: {len(df):,}")
    print(f"Fraud transactions: {int(y.sum()):,}")
    print(f"Legitimate transactions: {int((y == 0).sum()):,}")

    # Keep a held-out test set untouched during threshold selection.
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Use part of the training data to select a decision threshold.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=0.20,
        random_state=42,
        stratify=y_train_all
    )

    def build_model():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ])

    print("\nTraining baseline model...")
    validation_model = build_model()
    validation_model.fit(X_train, y_train)

    val_probabilities = validation_model.predict_proba(X_val)[:, 1]

    # Select the threshold that maximizes F1 on validation data only.
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        val_predictions = (val_probabilities >= threshold).astype(int)
        score = f1_score(y_val, val_predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    print(f"Selected threshold from validation set: {best_threshold:.2f}")

    # Retrain using all training data, without touching the test set.
    print("Training final model...")
    model = build_model()
    model.fit(X_train_all, y_train_all)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)

    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    average_precision = average_precision_score(
        y_test, test_probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    estimated_cost = (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "model": "Logistic Regression with StandardScaler",
        "threshold": best_threshold,
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_average_precision": float(average_precision),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_cost_assumption_inr": FALSE_POSITIVE_COST,
        "false_negative_cost_assumption_inr": FALSE_NEGATIVE_COST,
        "estimated_cost_inr": int(estimated_cost),
        "test_transactions": int(len(y_test))
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 score:         {f1:.4f}")
    print(f"Average precision:{average_precision:.4f}")
    print("\nConfusion matrix:")
    print(f"True negatives:   {tn}")
    print(f"False positives:  {fp}")
    print(f"False negatives:  {fn}")
    print(f"True positives:   {tp}")

    print("\nEstimated cost using illustrative assumptions:")
    print(f"₹{estimated_cost:,}")
    print("(This is a scenario estimate, not verified real-world savings.)")

    print("\nClassification report:")
    print(classification_report(
        y_test, test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    print(f"Model saved to:   {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
