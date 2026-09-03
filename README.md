# Detecting Abnormal Fuel Pricing Behavior in Mato Grosso

Unsupervised machine learning project for detecting unusual weekly gasoline pricing behavior in **Mato Grosso, Brazil**, using public data from the **Brazilian National Agency of Petroleum, Natural Gas and Biofuels (ANP)**.

> **Important:** an anomaly in this project means a statistically unusual observation. It is **not** evidence of fraud, price abuse, collusion, or any other illegal behavior.

![Anomaly score map](assets/11_mapa_anomaly_score_mt_2026.png)

## Project overview

The goal is to answer a simple question:

> **Can unsupervised machine learning identify unusual fuel pricing behavior when local market conditions and each station's own history are taken into account?**

Instead of treating a high price as automatically anomalous, the model uses a multidimensional representation of each **gas station × week** observation.

The pipeline combines:

- temporal feature engineering;
- local market benchmarks;
- Isolation Forest;
- Local Outlier Factor;
- One-Class SVM;
- Autoencoder;
- ensemble anomaly scoring;
- synthetic anomaly injection for controlled evaluation;
- PCA visualization;
- SHAP-based interpretation;
- municipal geospatial analysis.

## Dataset

The original ANP data covers weekly fuel price surveys. After filtering for Mato Grosso and gasoline, the notebook produced:

| Stage | Observations |
|---|---:|
| Mato Grosso, all fuels | 36,184 |
| Gasoline records | 9,506 |
| ML-ready observations | 8,679 |
| Training set — 2024/2025 | 6,769 |
| Test set — 2026 H1 | 1,910 |

The 2026 test period contains **138 gas stations across 7 municipalities**.

The train/test split is temporal rather than random:

```text
2024 ───────────── 2025 ─────────────│──────── 2026 H1
               TRAIN                 │          TEST
```

This design is closer to a real monitoring scenario: models learn from past behavior and score future observations.

## Feature engineering

Each row represents one gas station in one week.

The models use 15 features that describe three different dimensions of pricing behavior.

### Local market context

- current gasoline price;
- deviation from municipal median;
- municipal price percentile;
- deviation from Mato Grosso median;
- number of surveyed stations in the municipality.

### Station history

- 1-week price change;
- 4-week price change;
- deviation from the previous 4-week moving average;
- 4-week rolling volatility;
- 8-week rolling volatility;
- historical z-score.

### Seasonality

- sine/cosine encoding for month;
- sine/cosine encoding for week of year.

![Weekly gasoline price](assets/01_serie_preco_mt.png)

The historical series shows clear changes in the price level over time. This is one reason a fixed threshold such as *"anything above R$ X is anomalous"* would be inadequate.

## Models

Four unsupervised models were trained on 2024–2025 data.

### Isolation Forest

Detects observations that can be isolated from the rest of the feature space with relatively few tree splits.

### Local Outlier Factor

Measures whether an observation has substantially lower local density than its neighbors.

### One-Class SVM

Learns a nonlinear boundary around the region associated with historical observations.

### Autoencoder

A neural network learns to reconstruct historical feature vectors. Large reconstruction errors are treated as evidence that a new observation differs from learned patterns.

![Autoencoder training](assets/04_autoencoder_training.png)

The Autoencoder reached approximately **0.064 training MSE** and **0.093 validation MSE** after 150 epochs.

## Controlled evaluation with synthetic anomalies

Real ANP observations do not come with an `anomaly = 0/1` label.

To compare the algorithms quantitatively, the notebook injects **66 controlled synthetic anomalies** into the 2026 test period using artificial upward and downward price shocks. Features are then recalculated before scoring.

This benchmark does **not** imply that every unmodified observation is truly normal. It only provides a controlled experiment for comparing model sensitivity.

### Model performance

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **Isolation Forest** | **0.997** | **0.896** | **0.959** | 0.712 | **0.817** |
| Ensemble | 0.995 | 0.837 | 0.804 | 0.561 | 0.661 |
| One-Class SVM | 0.948 | 0.257 | 0.257 | **1.000** | 0.409 |
| LOF | 0.947 | 0.251 | 0.252 | 0.985 | 0.401 |
| Autoencoder | 0.904 | 0.158 | 0.163 | 0.561 | 0.253 |

![Model comparison](assets/05_model_comparison.png)

The **Isolation Forest was the strongest individual model**. It combined a very high ROC-AUC with a substantially better Precision-Recall profile than the alternatives.

LOF and One-Class SVM achieved very high recall, but their precision was close to 0.25 under the selected threshold. In practice, they were considerably more aggressive and generated many more false positives in the synthetic benchmark.

![Precision recall](assets/07_precision_recall.png)

## Ensemble scoring

Raw anomaly scores are not directly comparable across algorithms.

For each model, the notebook converts the score into an empirical percentile based on the training distribution:

```text
Isolation Forest ─┐
LOF ──────────────┤
One-Class SVM ────┼──> percentile scores ───> mean ───> ensemble score
Autoencoder ──────┘
```

The final threshold was approximately **0.9867**.

In the real 2026 dataset:

- **1,910** observations were scored;
- **39** were flagged by the ensemble;
- this corresponds to **2.04%** of the test observations.

## What do the anomalies look like?

PCA is used only for visualization — the models are trained on the full feature set.

![PCA anomalies](assets/09_pca_anomalies.png)

The first two principal components explain approximately **59.64%** of the variance.

Several flagged observations appear outside the main concentration of points, but there is also overlap between flagged and non-flagged observations. This is expected: anomaly detection in this project depends on the interaction among price level, local context, volatility and historical behavior.

## Geographic patterns

The ensemble produced the following anomaly rates among the municipalities represented in the ANP sample:

| Municipality | Anomaly rate | Maximum score | Observations |
|---|---:|---:|---:|
| **Sorriso** | **10.10%** | 0.996 | 99 |
| **Sinop** | **5.60%** | 0.999 | 250 |
| Várzea Grande | 1.79% | 0.995 | 392 |
| Alta Floresta | 1.11% | 0.992 | 180 |
| Cuiabá | 1.10% | 0.997 | 456 |
| Rondonópolis | 0.31% | 0.995 | 327 |
| Cáceres | 0.00% | 0.982 | 206 |

![Municipality anomaly ranking](assets/12_ranking_municipios_anomalia.png)

These rates should **not** be interpreted as rankings of misconduct or market quality. They are model outputs from a survey sample with different numbers of observations and station histories.

The geographic coverage is also an important limitation: the gasoline sample used in this notebook contains only **7 municipalities**, so most Mato Grosso municipalities have no observations in the analysis.

## Case study: an unusual station trajectory in Sinop

The highest ensemble score in 2026 was approximately **0.999**.

![Station case study](assets/13_case_study_top_station.png)

The selected observation had:

- price: **R$ 6.99/L**;
- municipal median: **R$ 6.89/L**;
- deviation from municipal median: only **+1.45%**;
- historical z-score: **2.21**;
- Isolation Forest percentile: **0.9999**;
- LOF percentile: **0.9973**;
- One-Class SVM percentile: **0.9996**;
- Autoencoder percentile: **0.9994**.

This is an important result.

The point is not extreme simply because its absolute price is high. The models are detecting a **combination of historical and multivariate behavior**.

Earlier in 2026, the same station had a weekly observation at **R$ 7.09/L**, following an **18.36% weekly increase**, with an extreme historical z-score. The subsequent trajectory remained unusual relative to its own previous behavior.

## Model interpretation

SHAP was applied to the Isolation Forest anomaly-score function to obtain an approximate global interpretation.

![SHAP Isolation Forest](assets/14_shap_isolation_forest.png)

The most influential variables included:

1. current price;
2. one-week price variation;
3. deviation from the recent moving average;
4. short-term rolling volatility;
5. historical z-score;
6. longer-term volatility;
7. number of surveyed competitors;
8. municipal price deviation.

This supports the main modeling idea: **anomalies are not defined by absolute price alone**.

> SHAP values here explain the model score; they should not be interpreted as causal effects on fuel prices.

## Main findings

- Isolation Forest was the best individual detector in the controlled synthetic benchmark.
- The ensemble retained excellent ranking ability but was more conservative at the selected 2% threshold.
- LOF and One-Class SVM were highly sensitive, but generated substantially more false positives.
- The Autoencoder learned the historical feature structure successfully, but reconstruction error was less discriminative than Isolation Forest.
- Sorriso and Sinop had the highest concentration of ensemble flags in the 2026 sample.
- Some of the strongest anomalies were driven by historical trajectory and volatility rather than large deviations from the current municipal median.

## Limitations

This project has several important limitations:

1. **Limited geographic coverage.** Only seven Mato Grosso municipalities appear in the gasoline sample used here.
2. **No confirmed real-world anomaly labels.** Model evaluation relies on controlled synthetic shocks.
3. **Operational anomaly threshold.** The 2% cutoff is a modeling choice, not a universal definition of abnormal pricing.
4. **No causal interpretation.** The models identify unusual patterns; they do not explain why prices changed.
5. **External drivers are not included.** Exchange rate, refinery prices, crude oil, freight, taxes and logistics could improve the feature space.
6. **Station-level results should not be interpreted as allegations.**

## Repository structure

```text
mt_fuel_anomaly/
│
├── README.md
├── mt_fuel_anomaly_ml_colab.ipynb
├── Relatorio_Anomalias_Precos_Combustiveis_MT.docx
│
└── assets/
    ├── 01_serie_preco_mt.png
    ├── 02_distribuicao_precos.png
    ├── 03_ranking_preco_municipios.png
    ├── 04_autoencoder_training.png
    ├── 05_model_comparison.png
    ├── 06_roc_models.png
    ├── 07_precision_recall.png
    ├── 08_model_agreement.png
    ├── 09_pca_anomalies.png
    ├── 10_mapa_preco_mediano_mt_2026.png
    ├── 11_mapa_anomaly_score_mt_2026.png
    ├── 12_ranking_municipios_anomalia.png
    ├── 13_case_study_top_station.png
    └── 14_shap_isolation_forest.png
```

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- GeoPandas
- Matplotlib
- SHAP
- ANP public fuel-price data
- IBGE municipal boundaries

## How to run

A detailed Portuguese analysis is available in [`Relatorio_Anomalias_Precos_Combustiveis_MT.docx`](Relatorio_Anomalias_Precos_Combustiveis_MT.docx).

The project was developed in **Google Colab**.

Open the notebook and run the cells from top to bottom. The notebook:

1. downloads the ANP datasets;
2. filters Mato Grosso gasoline records;
3. creates weekly station-level observations;
4. performs temporal feature engineering;
5. trains the four unsupervised models;
6. injects synthetic anomalies;
7. evaluates the models;
8. creates the ensemble;
9. downloads IBGE municipal boundaries;
10. generates the plots and maps.

## Possible next steps

- include ethanol and diesel;
- train municipality-specific detectors;
- add refinery, crude-oil and exchange-rate variables;
- include station brand and logistics features;
- optimize ensemble weights using the synthetic benchmark;
- perform rolling temporal retraining;
- deploy a Streamlit monitoring dashboard;
- automate weekly updates from ANP.

## Data sources

- **ANP** — Historical Fuel Price Series
- **IBGE** — Municipal geographic boundaries

---

### Why this project matters

The main takeaway is simple:

> **A price can be unusual without being the highest price in the market.**

By combining local benchmarks with each station's own temporal behavior, unsupervised machine learning can identify pricing patterns that would be missed by simple fixed thresholds.
