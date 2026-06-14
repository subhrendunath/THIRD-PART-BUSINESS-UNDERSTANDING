import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def load_data(data_dir="data"):
    """Load the churn dataset with features."""
    path = os.path.join(data_dir, "rfm_modeling_snapshot.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    return df


def prepare_splits(df):
    """Separate data into train, validation, and test sets."""
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
    """Extract features and target variable."""
    drop_columns = ["customer_id", "snapshot_date", "split", "churn_next_60d"]
    feature_columns = [col for col in df.columns if col not in drop_columns]
    if not feature_columns:
        raise ValueError("No feature columns available after dropping metadata columns.")
    return df[feature_columns], df["churn_next_60d"].astype(int)


def build_xgb_pipeline(df):
    """Build preprocessing and XGBoost model pipeline."""
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
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=1,
    )
    
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def evaluate_predictions(y_true, y_pred, y_proba, split_name):
    """Compute comprehensive evaluation metrics for churn classification."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_churn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_churn": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_churn": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "matthews_cc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
            "churn_rate": y_true.mean(),
            "predicted_churn_rate": y_pred.mean(),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
        }
    )
    
    print(f"\n{'='*70}")
    print(f"EVALUATION FOR {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    print(f"Sample size: {len(y_true):,}")
    print(f"\n--- CHURN CLASSIFICATION METRICS ---")
    print(f"Accuracy:                 {metrics['accuracy']:.4f}")
    print(f"F1-Score (Churn):         {metrics['f1_churn']:.4f}")
    print(f"Precision (Churn):        {metrics['precision_churn']:.4f}")
    print(f"Recall/Sensitivity (Churn): {metrics['sensitivity']:.4f}")
    print(f"Specificity (Non-Churn):  {metrics['specificity']:.4f}")
    print(f"\n--- CHURN PREVALENCE ---")
    print(f"Actual churn rate:        {metrics['churn_rate']:.4f}")
    print(f"Predicted churn rate:     {metrics['predicted_churn_rate']:.4f}")
    print(f"\n--- RANKING METRICS ---")
    print(f"ROC-AUC:                  {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:                   {metrics['pr_auc']:.4f}")
    print(f"Log Loss:                 {metrics['log_loss']:.4f}")
    print(f"\n--- AGREEMENT METRICS ---")
    print(f"Cohen's Kappa:            {metrics['cohen_kappa']:.4f}")
    print(f"Matthews CC:              {metrics['matthews_cc']:.4f}")
    print(f"\n--- CONFUSION MATRIX ---")
    print(f"True Negatives:  {tn:>6}    False Positives: {fp:>6}")
    print(f"False Negatives: {fn:>6}    True Positives:  {tp:>6}")
    print(f"\n--- FALSE RATES ---")
    print(f"False Positive Rate:      {metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate:      {metrics['false_negative_rate']:.4f}")
    print(f"\n--- PREDICTIVE VALUES ---")
    print(f"Positive Predictive Value (PPV): {metrics['ppv']:.4f}")
    print(f"Negative Predictive Value (NPV): {metrics['npv']:.4f}")
    
    print(f"\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0,
                                target_names=['Non-Churn', 'Churn']))
    return metrics


def compute_churn_suitability(y_true, y_proba):
    """Compute model suitability metrics for churn prediction."""
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    
    results = {}
    for percentile in [5, 10, 20, 25]:
        n = max(1, int(len(df) * percentile / 100))
        top_slice = df.iloc[:n]
        decile_rate = top_slice["y_true"].mean()
        overall_rate = df["y_true"].mean()
        lift = decile_rate / overall_rate if overall_rate > 0 else np.nan
        results[f"top_{percentile}pct"] = {
            "churn_rate": decile_rate,
            "overall_rate": overall_rate,
            "lift": lift,
            "count": n,
        }
    
    return results


def print_suitability_metrics(y_true, y_proba, split_name):
    """Print churn model suitability and lift metrics."""
    suitability = compute_churn_suitability(y_true, y_proba)
    print(f"\n{'='*70}")
    print(f"CHURN MODEL SUITABILITY - {split_name.upper()}")
    print(f"{'='*70}")
    for key, val in suitability.items():
        print(f"\n{key.upper()}:")
        print(f"  Overall churn rate:  {val['overall_rate']:.4f}")
        print(f"  Model churn rate:    {val['churn_rate']:.4f}")
        print(f"  Lift:                {val['lift']:.3f}x")
        print(f"  Sample size:         {val['count']:,}")


def summary_split_counts(subsets):
    """Print summary statistics for train/val/test splits."""
    print("\n" + "="*70)
    print("DATASET SPLIT SUMMARY")
    print("="*70)
    for split_name, split_df in subsets.items():
        churn_rate = split_df["churn_next_60d"].mean()
        print(f"  {split_name:12s}: {len(split_df):>5} rows, "
              f"churn rate = {churn_rate:.4f}, "
              f"churn count = {split_df['churn_next_60d'].sum():>5}")


def save_model(pipeline, model_path="model.pkl"):
    """Save the trained model pipeline to disk."""
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Model saved to: {model_path}")


def main():
    print("Starting XGBoost Churn Model Training and Evaluation...")
    
    df = load_data("data")
    subsets = prepare_splits(df)
    summary_split_counts(subsets)

    train_df = subsets["train"]
    val_df = subsets["validation"]
    test_df = subsets["test"]

    pipeline = build_xgb_pipeline(df)

    X_train, y_train = create_feature_matrix(train_df)
    X_val, y_val = create_feature_matrix(val_df)
    X_test, y_test = create_feature_matrix(test_df)

    print("\n" + "="*70)
    print("TRAINING XGBoost MODEL...")
    print("="*70)
    pipeline.fit(X_train, y_train)
    print("✓ Model training complete")

    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_predictions(y_train, y_train_pred, y_train_proba, "train")
    val_metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba, "validation")
    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba, "test")

    print_suitability_metrics(y_val, y_val_proba, "validation")
    print_suitability_metrics(y_test, y_test_proba, "test")

    save_model(pipeline, "model.pkl")
    
    print("\n" + "="*70)
    print("XGBoost Churn Model Complete!")
    print("="*70)
    print("Outputs:")
    print("  - model.pkl: Trained XGBoost pipeline ready for predictions")
    print("  - Evaluation metrics printed above for train/val/test splits")


if __name__ == "__main__":
    main()
