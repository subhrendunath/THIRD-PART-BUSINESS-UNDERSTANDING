# Model Card: Customer Churn Prediction

## Overview

This document describes a binary classification model saved as `model.pkl` used to predict the probability that a customer will churn within the next 60 days. The model supports a retention program that prioritizes outreach and retention offers. This card summarizes intended use, training and validation data, observed performance on the validation split, known limitations, ethical risks, and recommended monitoring and governance measures.

## Intended Use

- Primary use: prioritize customers for retention outreach by estimating P(churn within 60 days). The score is intended to be combined with business rules (e.g., capacity, customer value) to select an outreach list.
- Decision context: marketing and retention teams use model scores to allocate a finite budget of interventions (coupons, calls, emails). The model is not a replacement for human review for high-value customers.
- Not intended for: denial of service, permanent customer exclusion, credit or legal decisions, or any use that would adversely affect customers without human oversight.

Users should treat the model as a decision-support tool and always test actions using small controlled experiments (A/B tests) before broad rollout.

## Data (training / validation)

- Source: project data files in `data/` (notably `data/churn_labels.csv` and `data/splits/`). The validation split is in `data/splits/validation.csv`.
- Label definition: `churn_next_60d` — a binary indicator whether the customer churned in the 60 days following the snapshot date.
- Features: behavioral and RFM-style features including `recency_days`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `avg_discount_pct_180d`, `avg_rating_180d`, `sessions_30d`, `product_views_30d`, `cart_adds_30d`, loyalty tier, marketing consent, acquisition channel, and related fields. Feature engineering and preprocessing were applied before model training (pipeline persisted in `model.pkl`).
- Data caveats: the dataset reflects a specific time window and product mix; there may be seasonality, cohort effects, and business changes not captured in the historical data.

## Performance (validation)

Summary metrics computed on the validation split using a decision threshold of 0.60:

- Confusion matrix (validation, threshold 0.60): TN = 164, FP = 25, FN = 43, TP = 104
- Precision: 0.806
- Recall: 0.707
- F1 score: 0.754

Class prevalence (validation): approximately 47% of customers labeled as churn in `data/churn_labels.csv` (baseline churn rate ≈ 0.47). This high base rate influences thresholding decisions and expected outreach volume.

Cost-sensitive evaluation example: a simple profit sweep using Benefit = 100 (value from a successfully retained customer) and Cost = 5 (intervention cost) showed that under those assumptions contacting more customers can be profitable; a formal profit-based threshold should be computed using the real intervention cost and estimated retention lift.

Notes:
- Reported metrics reflect a specific threshold and validation split. Different thresholds or business objectives (maximize precision vs maximize recall vs profit) will change recommended operating points.
- The reported performance is for held-out validation only; deployment performance will vary and should be measured in live experiments.

## Top features driving predictions

Analysis of the model and surrogate explanations indicate the strongest predictive signals are:

- `recency_days` (days since last purchase/visit): long recency increases predicted churn probability.
- `sessions_30d` and `frequency_180d`: recent engagement and purchase frequency reduce churn probability.
- `monetary_180d`: higher spending reduces predicted churn and raises priority for retention when positive.
- `return_rate_180d` and `avg_rating_180d`: returns and low ratings are risk signals.
- `marketing_consent` and `loyalty_tier`: operational constraints (no consent) and loyalty tier moderate practical actionability.

Business interpretation: recency and engagement are dominant. The model tends to flag customers with long gaps since last activity; conversely it can underpredict churn for customers with high recent engagement but unobserved negative experiences.

## Limitations

- Feature coverage: the model relies on observed transactional and engagement features. It does not include certain causal drivers of churn such as product delivery failures, customer service incidents beyond ticket counts, competitor offers, or personal events.
- Label timing and noise: churn labels reflect events after the snapshot and may capture short-term churn due to external causes; label noise reduces achievable performance.
- Dataset shift: model performance will degrade if customer behavior, product assortment, pricing, or marketing policies change. Seasonality and new acquisition channels can also introduce drift.
- Threshold sensitivity: the operating threshold determines outreach volume and tradeoffs between precision and recall. A single static threshold may not suit all campaigns or customer segments.
- Surrogate explanations: when SHAP could not be computed in this environment, a linear surrogate was fitted. These surrogate contributions approximate but do not replace per-model, per-customer explainability methods.

## Ethical and business risks

- False positives: contacting customers incorrectly identified as high-risk can incur cost, fatigue, and reputational risk. Over-contacting can reduce long-term engagement.
- False negatives: failing to identify high-risk customers (especially high-value ones) can cause lost revenue and missed retention opportunities.
- Privacy and consent: models that recommend outreach must respect `marketing_consent`. Automated outreach to customers without consent is both unethical and often illegal under privacy regulations.
- Disparate impact: features such as acquisition channel, city tier, or demographic proxies could lead to unequal treatment. Evaluate outcomes across key groups (age, region, loyalty tier) to detect and mitigate biased behavior.

Mitigations:
- Use the model scores with business rules (e.g., always protect Platinum customers; require marketing consent). Implement caps on contact frequency and segment-specific thresholds. Run A/B tests before scaling.
- Maintain an appeals or human review path for high-value decisions.

## Monitoring and maintenance

Before and after deployment implement monitoring for:

- Data quality: detect missing columns, unexpected data types, and sudden shifts in feature distributions (population-level drift).
- Model performance drift: track precision/recall, predicted positive rate, and calibration over time using labeled holdout or periodic labeling windows.
- Business metrics: monitor intervention cost, retention lift (measured via controlled experiments), and ROI per cohort.
- Fairness metrics: monitor performance and outreach rates across sensitive or business-relevant subgroups.

Operational recommendations:

- Retrain cadence: retrain the model when performance drops beyond a threshold (for example, >5% absolute drop in precision or recall) or after major business changes (pricing, channel mix). Initially consider monthly retraining while gathering live labels, then adjust cadence to observed stability.
- Threshold management: select thresholds using a profit-aware objective that includes estimated intervention cost and lift. Maintain a small “pilot” audience for each new threshold selection before full rollout.
- Logging and auditability: log model inputs, scores, and actions taken for a minimum retention period to allow post-hoc analysis and compliance audits. Ensure logs exclude or mask PII.

## Recommended next steps

1. Run SHAP explanations on an environment without the DLL restriction to produce per-customer feature attributions for operational transparency.
2. Design and run an A/B experiment that measures incremental retention lift and computes true ROI, then re-evaluate threshold and outreach policy.
3. Add support event features (detailed ticket text or cancellation signals) to improve recall for sudden churns among high-value customers.
4. Implement monitoring dashboards for data drift, performance, and fairness, and set alert thresholds.

## Contact and provenance

Model artifact: `model.pkl` in repository root. Key analysis artifacts created during model evaluation are in `analysis/` (including `error_analysis_examples.csv` and `shap_alternative_explanations.csv`).

Prepared by the model development team in this repository. For questions about model usage, data, or to request more detailed fairness analysis, contact the project owner maintained in repository metadata.

---

This model card is a living document and should be updated when the model, data, or business use case changes.
