# THIRD-PART-BUSINESS-UNDERSTANDING
IITP MASAI SEM-3-PART #3 PROJECT

## Project Writing Process — Illustration

This section describes the practical writing process used to create the project deliverables: code, analysis artifacts, visualizations, and documentation. The goal is to make the project's authoring transparent and reproducible for reviewers and collaborators. The process is organized into planning, data exploration, modeling, evaluation, interpretation, documentation, and handoff.

1. Planning and scope definition

- Define business question: reduce customer churn by detecting customers at risk within 60 days and prioritizing retention outreach. Clarify success metrics (precision, recall, ROI) and constraints (marketing consent, contact capacity).
- Establish deliverables: a reproducible modeling pipeline (`churn_model.py`, `baseline_churn_model.py`), evaluation scripts (`visualize_churn_model.py`, `visualizations/`), a model artifact (`model.pkl`), and a final report (`VISUALIZATION_REPORT.md`).
- Create a lightweight project map listing directories and responsibilities (data preparation, modeling, visualization, documentation).

2. Data understanding and exploratory analysis

- Inventory data files under `data/` and examine `DATA_DICTIONARY.md` to confirm feature semantics. Initial checks focused on completeness, label definition `churn_next_60d`, and split strategy under `data/splits/`.
- Produce quick EDA outputs with `eda_customer_behavior.py` to visualize distributions, missingness, and correlations. Visual outputs are saved to `visualizations/` for review and iteration.
- Record data quality issues and corrections in `data/data_quality_report_files/` and document decisions (e.g., how to treat missing loyalty tiers or categorical levels).

3. Data preparation and feature engineering

- Implement a preprocessing pipeline that handles missing values, encodes categorical variables, and scales numeric inputs. Keep preprocessing steps in code so they are reproducible and persistable as part of `model.pkl` when using a scikit-learn `Pipeline`.
- Create derived features that proved useful: RFM-style metrics (`frequency_180d`, `monetary_180d`, `recency_days`), engagement features (`sessions_30d`, `product_views_30d`), and recent support activity (ticket counts). Log feature transformations in comments and in the notebook used for experiments.

4. Modeling iterations

- Start with a simple baseline model (`baseline_churn_model.py`) to establish a benchmark. Iterate toward more complex models and pipelines in `churn_model.py` while tracking metric improvements.
- Use cross-validation and holdout splits (train/validation/test under `data/splits/`) to avoid information leakage. Track versions of the model and hyperparameters; save the final production candidate as `model.pkl`.

5. Evaluation and error analysis

- Evaluate using confusion matrices, precision, recall, F1, and business-oriented profit sweeps. The notebook and scripts compute threshold sweeps to align the model with outreach costs and capacity.
- Perform targeted error analysis focusing on false positives and false negatives. Create `analysis/error_analysis_examples.csv` to capture representative cases and generate approximate feature attributions (stored in `analysis/shap_alternative_explanations.csv`) to explain model decisions.

6. Interpretation and business translation

- Translate model outputs to actionable policies: recommended operating threshold, gating rules (protect high-value customers), and consent checks. Produce `model_card.md` summarizing intended use, limitations, ethical risks, and monitoring needs.
- Present compact, human-readable summaries and the top contributing features in `analysis/top_feature_explanations.txt` so non-technical stakeholders can understand drivers of churn and implications for campaigns.

7. Documentation and reproducibility

- Keep scripts single-purpose and parameterized so others can re-run steps. Use relative paths and a `data/` folder layout to ensure reproducibility. Document runtime environment requirements; a virtual environment `.venv/` is present in the workspace for local development.
- Log all generated artifacts (models, CSVs, plots) in `analysis/` and `visualizations/` so reviewers can trace results back to specific code versions.

8. Collaboration and version control

- Use Git for incremental commits; meaningful commit messages were used when adding analysis artifacts (`analysis/error_analysis_examples.csv`, `analysis/shap_alternative_explanations.csv`). Keep code and docs together so reviewers can run the pipeline end-to-end.

9. Handoff and monitoring recommendations

- For operational handoff, include the `model_card.md` and the `analysis/` folder with examples and explanations. Document contact points and next steps to maintain the model (monitoring thresholds, retraining cadence, A/B testing plans).

10. Writing and polishing notes

- Draft content in short, iterative passes: quick prototypes for code and plots, then consolidate results into permanent scripts and markdown documentation. Keep prose focused on reproducibility and decision-making, not only on numbers.
- When preparing the final report, emphasize clarity: define each metric, explain thresholds in business terms, and provide exact file paths to find scripts and artifacts.

This writing process ensures that the project is both reproducible and actionable: code artifacts are runnable, analyses are documented, and business recommendations are traceable to data and model outputs. For a quick start, run `visualize_churn_model.py` to view the main evaluation plots and inspect the `analysis/` folder for concrete examples used during error analysis.

## What we did (churn prediction pipeline)

This project implemented a reproducible churn prediction pipeline with the objective of identifying customers at risk of churning within 60 days and prioritizing them for retention outreach. The work proceeded through clearly defined stages: data assembly and validation, feature engineering, model iteration, evaluation against business metrics, and interpretability and error analysis. Below is a concise summary of each stage and the concrete artifacts produced.

Data assembly and validation: We collected customer transactional and behavioral data into `data/` and verified the label `churn_next_60d` in `data/churn_labels.csv`. Splits for train/validation/test were prepared under `data/splits/` to prevent information leakage. Initial checks identified missing values, categorical level inconsistencies, and skewed distributions; these were recorded in the project's data quality notes and addressed in preprocessing.

Feature engineering and preprocessing: We built a consistent preprocessing pipeline to handle numeric scaling, imputation, and categorical encoding so experiments are reproducible. Key engineered features include recency (`recency_days`), frequency (`frequency_180d`), monetary (`monetary_180d`), recent engagement (`sessions_30d`, `product_views_30d`), and support-related counts (`ticket_count_90d`). Preprocessing steps and the fitted transformers are persisted within the production pipeline so `model.pkl` can accept raw rows from the dataset format.

Modeling iterations: A baseline model established a performance floor (`baseline_churn_model.py`). We iterated models and pipelines in `churn_model.py`, optimizing for practical metrics (precision at a target recall, and business profit). The final production candidate is saved as `model.pkl` in the repository root. Models were validated using the held-out validation split to estimate real-world behavior before any pilot deployment.

Evaluation and operating point selection: Core evaluation used confusion matrices, precision, recall, F1, and threshold sweeps. We also ran a cost-sensitive sweep (example Benefit=100, Cost=5) to illustrate profit-aware thresholding and to demonstrate how the outreach policy should be selected using business parameters. For a conservative operational starting point we examined thresholds that control outreach volume while preserving acceptable precision.

Error analysis and interpretability: To understand failures, we extracted representative false positives and false negatives into `analysis/error_analysis_examples.csv`. We attempted model-appropriate SHAP explanations; where SHAP was blocked by environment constraints, we produced a surrogate linear-explanation file `analysis/shap_alternative_explanations.csv`. These artifacts help identify that `recency_days` heavily drives positive risk signals while strong engagement features (sessions, frequency, monetary) counteract risk — a pattern that explains many false positives and false negatives.

Documentation and model card: We produced `model_card.md` summarizing intended use, data provenance, performance, limitations, ethical risks, and monitoring suggestions. We also added `analysis/top_feature_explanations.txt` to provide a short, non-technical summary of the most important features for stakeholders.

Artifacts and reproducibility: Key artifacts created are `model.pkl` (model artifact), evaluation and visualization scripts (`visualize_churn_model.py`, `visualizations/`), and the `analysis/` folder containing CSV examples and explanation outputs. All code is written to accept relative paths and a contained `data/` folder so reviewers can re-run experiments locally.

Recommendations: Before production rollout, run a small pilot (A/B test) to measure incremental retention lift and refine cost assumptions. Add additional signals that capture customer experience (support ticket text, cancelled subscriptions) to reduce false negatives among high-value customers. Establish monitoring for data drift, model performance, and fairness metrics; define a retraining cadence and guardrails for outreach volume.

This summary captures the practical steps taken in the churn prediction project and points to the files and scripts you can run to reproduce each stage.

