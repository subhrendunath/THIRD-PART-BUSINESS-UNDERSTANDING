import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_dir="data"):
    path = os.path.join(data_dir, "rfm_modeling_snapshot.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    return df


def prepare_splits(df):
    required_columns = {"split", "churn_next_60d"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    split_map = {
        "train": "train",
        "validation": "validation",
        "val": "validation",
        "test": "test",
    }
    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower().map(split_map).fillna(df["split"].str.lower())

    subsets = {}
    for split_name in ["train", "validation", "test"]:
        subset = df[df["split"] == split_name].reset_index(drop=True)
        subsets[split_name] = subset
    return subsets


def create_feature_matrix(df):
    drop_columns = ["customer_id", "snapshot_date", "split", "churn_next_60d"]
    feature_columns = [col for col in df.columns if col not in drop_columns]
    if not feature_columns:
        raise ValueError("No feature columns available after dropping metadata columns.")
    return df[feature_columns], df["churn_next_60d"].astype(int)


def build_pipeline(df):
    X, _ = create_feature_matrix(df)
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def evaluate_predictions(y_true, y_pred, y_proba, split_name):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_churn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
            "churn_rate": y_true.mean(),
            "predicted_churn_rate": y_pred.mean(),
        }
    )
    print(f"\n=== Evaluation for {split_name.upper()} split ===")
    print(f"Rows: {len(y_true)}")
    print(f"Churn prevalence: {metrics['churn_rate']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision (churn): {metrics['precision_churn']:.3f}")
    print(f"Recall (churn): {metrics['recall_churn']:.3f}")
    print(f"F1 (churn): {metrics['f1_churn']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"Predicted churn rate: {metrics['predicted_churn_rate']:.3f}")
    print("Confusion matrix:")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return metrics


def compute_top_decile_lift(y_true, y_proba):
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    top_decile = df.iloc[: max(1, int(len(df) * 0.1))]
    decile_rate = top_decile["y_true"].mean()
    overall_rate = df["y_true"].mean()
    lift = decile_rate / overall_rate if overall_rate > 0 else np.nan
    return {
        "top_decile_rate": decile_rate,
        "overall_rate": overall_rate,
        "lift_top_10pct": lift,
        "top_decile_size": len(top_decile),
    }


def summary_split_counts(subsets):
    print("Dataset split counts:")
    for split_name, split_df in subsets.items():
        churn_rate = split_df["churn_next_60d"].mean()
        print(f"  {split_name}: {len(split_df)} rows, churn rate = {churn_rate:.3f}")


def save_splits(subsets, data_dir="data"):
    split_dir = os.path.join(data_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for split_name, split_df in subsets.items():
        path = os.path.join(split_dir, f"{split_name}.csv")
        split_df.to_csv(path, index=False)
    print(f"Saved split files to: {split_dir}")


def main():
    df = load_data("data")
    subsets = prepare_splits(df)
    summary_split_counts(subsets)
    save_splits(subsets, "data")

    train_df = subsets["train"]
    val_df = subsets["validation"]
    test_df = subsets["test"]

    pipeline = build_pipeline(df)

    X_train, y_train = create_feature_matrix(train_df)
    X_val, y_val = create_feature_matrix(val_df)
    X_test, y_test = create_feature_matrix(test_df)

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    evaluate_predictions(y_train, y_train_pred, y_train_proba, "train")
    evaluate_predictions(y_val, y_val_pred, y_val_proba, "validation")
    evaluate_predictions(y_test, y_test_pred, y_test_proba, "test")

    print("\n=== Churn lift / suitability metrics ===")
    for name, y_true, y_proba in [
        ("validation", y_val, y_val_proba),
        ("test", y_test, y_test_proba),
    ]:
        lift = compute_top_decile_lift(y_true, y_proba)
        print(f"\n{name.upper()} top-decile lift:")
        print(f"  Overall churn rate: {lift['overall_rate']:.3f}")
        print(f"  Top 10% predicted churn rate: {lift['top_decile_rate']:.3f}")
        print(f"  Lift in top decile: {lift['lift_top_10pct']:.3f}")

    print("\nBaseline model pipeline created and evaluated."
          " The training/validation/test separation is saved under data/splits.")


if __name__ == "__main__":
    main()
