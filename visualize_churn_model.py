import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, matthews_corrcoef,
    cohen_kappa_score, log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
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


def plot_confusion_matrices(results_dict, output_dir="visualizations"):
    """Plot confusion matrices for all splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Confusion Matrices Across Splits', fontsize=16, fontweight='bold')
    
    splits = ['train', 'validation', 'test']
    
    for idx, (ax, split) in enumerate(zip(axes, splits)):
        y_pred = results_dict[split]['y_pred']
        y_true = results_dict[split]['y_true']
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(f'{split.capitalize()} Split (n={len(y_true):,})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '01_confusion_matrices.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_roc_curves(results_dict, output_dir="visualizations"):
    """Plot ROC curves for all splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    splits = ['train', 'validation', 'test']
    colors = ['blue', 'green', 'red']
    
    for split, color in zip(splits, colors):
        y_true = results_dict[split]['y_true']
        y_proba = results_dict[split]['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, color=color, lw=2.5, 
               label=f'{split.capitalize()} (AUC={roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
    ax.set_title('ROC Curves - Churn Prediction Model', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '02_roc_curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_precision_recall_curves(results_dict, output_dir="visualizations"):
    """Plot Precision-Recall curves for all splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    splits = ['train', 'validation', 'test']
    colors = ['blue', 'green', 'red']
    
    for split, color in zip(splits, colors):
        y_true = results_dict[split]['y_true']
        y_proba = results_dict[split]['y_proba']
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, color=color, lw=2.5,
               label=f'{split.capitalize()} (AUC={pr_auc:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='bold', fontsize=11)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=11)
    ax.set_title('Precision-Recall Curves - Churn Prediction Model', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '03_precision_recall_curves.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_metrics_comparison(results_dict, output_dir="visualizations"):
    """Plot comparison of key metrics across splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Performance Metrics Comparison Across Splits', fontsize=16, fontweight='bold')
    
    splits = ['train', 'validation', 'test']
    metrics = ['accuracy', 'precision_churn', 'recall_churn', 'f1_churn', 'roc_auc', 'pr_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    
    for idx, (ax, metric, label) in enumerate(zip(axes.flat, metrics, metric_labels)):
        values = [results_dict[split]['metrics'][metric] for split in splits]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(splits, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylim([0, 1.05])
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '04_metrics_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_churn_distribution(results_dict, output_dir="visualizations"):
    """Plot actual vs predicted churn distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Churn Distribution: Actual vs Predicted', fontsize=16, fontweight='bold')
    
    splits = ['train', 'validation', 'test']
    
    for idx, (ax, split) in enumerate(zip(axes, splits)):
        y_true = results_dict[split]['y_true']
        y_pred = results_dict[split]['y_pred']
        
        actual_rate = y_true.mean()
        predicted_rate = y_pred.mean()
        
        x = ['Actual', 'Predicted']
        y = [actual_rate, predicted_rate]
        colors_dist = ['#3498db', '#e74c3c']
        
        bars = ax.bar(x, y, color=colors_dist, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, value in zip(bars, y):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylim([0, max(y) * 1.2])
        ax.set_ylabel('Churn Rate', fontweight='bold')
        ax.set_title(f'{split.capitalize()} Split (n={len(y_true):,})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '05_churn_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_lift_analysis(results_dict, output_dir="visualizations"):
    """Plot model lift analysis at various percentiles."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Lift Analysis', fontsize=16, fontweight='bold')
    
    # Validation set
    y_true_val = results_dict['validation']['y_true']
    y_proba_val = results_dict['validation']['y_proba']
    
    df_val = pd.DataFrame({"y_true": y_true_val, "y_proba": y_proba_val})
    df_val = df_val.sort_values("y_proba", ascending=False).reset_index(drop=True)
    
    percentiles = [5, 10, 20, 25, 30, 40, 50]
    lift_vals = []
    
    for percentile in percentiles:
        n = max(1, int(len(df_val) * percentile / 100))
        top_slice = df_val.iloc[:n]
        decile_rate = top_slice["y_true"].mean()
        overall_rate = df_val["y_true"].mean()
        lift = decile_rate / overall_rate if overall_rate > 0 else np.nan
        lift_vals.append(lift)
    
    axes[0].plot(percentiles, lift_vals, marker='o', linewidth=2.5, markersize=8, color='#2ecc71')
    axes[0].axhline(y=1, color='k', linestyle='--', linewidth=1.5, label='Baseline (Lift=1)')
    axes[0].fill_between(percentiles, 1, lift_vals, alpha=0.2, color='#2ecc71')
    axes[0].set_xlabel('Top Percentile (%)', fontweight='bold')
    axes[0].set_ylabel('Lift', fontweight='bold')
    axes[0].set_title('Validation Set - Lift at Different Percentiles', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Test set
    y_true_test = results_dict['test']['y_true']
    y_proba_test = results_dict['test']['y_proba']
    
    df_test = pd.DataFrame({"y_true": y_true_test, "y_proba": y_proba_test})
    df_test = df_test.sort_values("y_proba", ascending=False).reset_index(drop=True)
    
    lift_vals_test = []
    
    for percentile in percentiles:
        n = max(1, int(len(df_test) * percentile / 100))
        top_slice = df_test.iloc[:n]
        decile_rate = top_slice["y_true"].mean()
        overall_rate = df_test["y_true"].mean()
        lift = decile_rate / overall_rate if overall_rate > 0 else np.nan
        lift_vals_test.append(lift)
    
    axes[1].plot(percentiles, lift_vals_test, marker='s', linewidth=2.5, markersize=8, color='#e74c3c')
    axes[1].axhline(y=1, color='k', linestyle='--', linewidth=1.5, label='Baseline (Lift=1)')
    axes[1].fill_between(percentiles, 1, lift_vals_test, alpha=0.2, color='#e74c3c')
    axes[1].set_xlabel('Top Percentile (%)', fontweight='bold')
    axes[1].set_ylabel('Lift', fontweight='bold')
    axes[1].set_title('Test Set - Lift at Different Percentiles', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '06_lift_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def plot_feature_importance(pipeline, X_train, output_dir="visualizations", top_n=15):
    """Plot top N feature importances from the XGBoost model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = []
    
    # Get numeric features
    numeric_features = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
    feature_names.extend(numeric_features)
    
    # Get categorical features (one-hot encoded)
    cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_transformer.get_feature_names_out()
    feature_names.extend(cat_features)
    
    # Get feature importances
    model = pipeline.named_steps['model']
    importances = model.feature_importances_
    
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors, edgecolor='black', linewidth=1)
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontweight='bold', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '07_feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()


def evaluate_predictions(y_true, y_pred, y_proba):
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
    metrics.update({
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "churn_rate": y_true.mean(),
        "predicted_churn_rate": y_pred.mean(),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
    })
    
    return metrics


def print_metrics(metrics, split_name):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"EVALUATION FOR {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Churn): {metrics['f1_churn']:.4f}")
    print(f"Precision (Churn): {metrics['precision_churn']:.4f}")
    print(f"Recall/Sensitivity (Churn): {metrics['sensitivity']:.4f}")
    print(f"Specificity (Non-Churn): {metrics['specificity']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")


def main():
    print("Starting XGBoost Churn Model Training with Visualization...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Load and prepare data
    df = load_data("data")
    subsets = prepare_splits(df)
    
    print("="*70)
    print("DATASET SPLIT SUMMARY")
    print("="*70)
    for split_name, split_df in subsets.items():
        churn_rate = split_df["churn_next_60d"].mean()
        print(f"  {split_name:12s}: {len(split_df):>5} rows, "
              f"churn rate = {churn_rate:.4f}, "
              f"churn count = {split_df['churn_next_60d'].sum():>5}")

    # Build and train pipeline
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

    # Make predictions
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate predictions
    train_metrics = evaluate_predictions(y_train, y_train_pred, y_train_proba)
    val_metrics = evaluate_predictions(y_val, y_val_pred, y_val_proba)
    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_proba)

    print_metrics(train_metrics, "train")
    print_metrics(val_metrics, "validation")
    print_metrics(test_metrics, "test")

    # Store results for visualization
    results_dict = {
        'train': {'y_true': y_train, 'y_pred': y_train_pred, 'y_proba': y_train_proba, 'metrics': train_metrics},
        'validation': {'y_true': y_val, 'y_pred': y_val_pred, 'y_proba': y_val_proba, 'metrics': val_metrics},
        'test': {'y_true': y_test, 'y_pred': y_test_pred, 'y_proba': y_test_proba, 'metrics': test_metrics},
    }

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70)
    
    plot_confusion_matrices(results_dict)
    plot_roc_curves(results_dict)
    plot_precision_recall_curves(results_dict)
    plot_metrics_comparison(results_dict)
    plot_churn_distribution(results_dict)
    plot_lift_analysis(results_dict)
    plot_feature_importance(pipeline, X_train)

    # Save model
    joblib.dump(pipeline, "model.pkl")
    print(f"\n✓ Model saved to: model.pkl")

    print("\n" + "="*70)
    print("XGBoost Churn Model Complete!")
    print("="*70)
    print("Visualization files saved in: ./visualizations/")
    print("  - 01_confusion_matrices.png")
    print("  - 02_roc_curves.png")
    print("  - 03_precision_recall_curves.png")
    print("  - 04_metrics_comparison.png")
    print("  - 05_churn_distribution.png")
    print("  - 06_lift_analysis.png")
    print("  - 07_feature_importance.png")


if __name__ == "__main__":
    main()
