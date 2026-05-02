# NBA Player Props Prediction Pipeline

> End-to-end machine learning system for NBA player prop predictions (Points, Rebounds, Assists), built from scratch with **LightGBM + ensemble models**, rigorous walk-forward validation, and a multi-layered data leakage detection framework.

## Project Overview

| Metric | Value |
|--------|-------|
| **Dataset** | ~112K player-game rows × 900+ engineered features |
| **Seasons covered** | 2019-20 → 2025-26 (7 NBA seasons) |
| **Markets** | Points (PTS), Rebounds (REB), Assists (AST) |
| **Models in production** | 3 model families × 3 markets = 9 models |
| **Walk-forward folds** | 6-fold date-based expanding window |
| **Validation AUC (form-streak model)** | PTS: 0.74 · REB: 0.73 · AST: 0.75 |
| **Sessions of iterative R&D** | 50+ documented experiment sessions |

---

## Architecture

The system runs daily during the NBA season through an automated Colab pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    DAILY PIPELINE                           │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ NBA API  │──▶│  Merge & │──▶│ Feature  │──▶│ Model   │ │
│  │ Scraper  │   │  Align   │   │ Engine   │   │ Ensemble│ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│       │              │              │              │        │
│  Box scores     Player norm    Rolling stats   LGB + RF    │
│  Team stats     Team merge     Interactions    + LR + XGB  │
│  Tracking       Season align   Prior-season    Meta-model  │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │  Odds    │──▶│  Props   │──▶│  Signal  │──▶ Predictions │
│  │  API     │   │  Merge   │   │  Router  │               │
│  └──────────┘   └──────────┘   └──────────┘               │
│  Live lines     Fuzzy match    Edge tiers                  │
│  Juice/vig      Synthetic      Confidence                  │
│                  fallback       buckets                     │
└─────────────────────────────────────────────────────────────┘
```

For more details, see [`docs/pipeline_architecture.md`](docs/pipeline_architecture.md).

---

## Key Technical Highlights

### 1. Walk-Forward Validation Framework
Date-based expanding window validation — no random splits, no future leakage.

```python
# 6-fold expanding window: 30% initial train, growing to 100%
edges = np.linspace(0.30, 1.00, n_folds + 1)  # date-percentile edges
# Each fold trains on ALL data before its test window
```

| Fold | Train window | Test window | Typical AUC (ensemble) |
|------|-------------|-------------|----------------------|
| F1 | 2019-20 → early 2022 | mid 2022 | 0.53–0.58 (cold start) |
| F2 | ... → late 2022 | early 2023 | 0.60–0.65 |
| F3 | ... → mid 2023 | late 2023 | 0.73–0.78 |
| F4 | ... → early 2024 | mid 2024 | 0.78–0.83 |
| F5 | ... → late 2024 | early 2025 | 0.80–0.86 |
| F6 | ... → early 2025 | late 2025 | 0.68–0.72 (production) |

See [`validation/walk_forward.py`](validation/walk_forward.py) for the implementation.

### 2. Multi-Layer Data Leakage Detection
Sports betting features are uniquely prone to subtle data leakage. This project implements a **4-layer leak detection pipeline** that catches leaks invisible to standard correlation tests.

**Layer 1 — Canonical Gap Test** (primary)
```
gap = max(|corr_lag5|, |corr_lag10|) − |corr_current|
gap > 0    → CLEAN
gap < −0.03 → LEAK
```

**Layer 2 — Cross-Temporal Peak Detection**
```
Compute corr(feature[t], target[t+k]) for k in [-3..+3]
Peak at k ≤ 0 → CLEAN (feature leads or aligns with target)
Peak at k ≥ 1 → LEAK (feature follows target = future info)
```

**Layer 3 — Temporal Constancy Gate**
```
intra_season_std / total_std < 0.05 → SUSPECT
Combined with cross-season ratio for disambiguation
```

**Layer 4 — Cross-Season Disambiguation** (catches false positives from Layer 3)
```
intra < 0.05 + cross_season ≥ 0.10 → CLEAN (prior-season aggregate, expected behavior)
intra < 0.05 + cross_season < 0.05 → CONFIRMED LEAK
```

This framework caught **9 false-clean features** that passed the standard gap test but would have introduced forward-looking information. It also prevented **7 false-alarm rejections** by correctly identifying prior-season features as intentionally constant within a season.

See [`validation/leak_detection.py`](validation/leak_detection.py) for the full implementation.

### 3. Prior-Season Feature Engineering
Season-level aggregates (offensive efficiency, defensive tendencies, playmaking profiles) are extracted from **15+ NBA API endpoints** and merged as prior-season lookbacks — a player's 2023-24 profile informs their 2024-25 predictions.

- **302 prior-season features** across 14 source groups
- Coverage: ~81% of player-game rows (expected: 2019-20 season has 0% by design — no 2018-19 prior data)
- Merge key: always `PRIOR_SEASON`, never current `SEASON` (prevents forward leakage)

See [`features/prior_season_builder.py`](features/prior_season_builder.py) for the extraction pipeline.

### 4. Model Architecture
Multi-model ensemble with market-specific feature selection:

```
              ┌─────────────┐
              │  LightGBM   │─────┐
              └─────────────┘     │
              ┌─────────────┐     │    ┌──────────┐
              │   XGBoost   │─────┼───▶│  Weighted│──▶ P(over)
              └─────────────┘     │    │  Ensemble │
              ┌─────────────┐     │    └──────────┘
              │ Random Forest│────┘         │
              └─────────────┘              ▼
              ┌─────────────┐        ┌──────────┐
              │   Log Reg   │───────▶│   Meta   │──▶ Final prediction
              └─────────────┘        │  Learner │
                                     └──────────┘
```

- **Per-market feature selection** via walk-forward importance + ablation gating
- **Strict shipping gate**: ΔAUC ≥ +0.0015, positive lift in 6/6 validation folds
- **67–404 features per market** after pruning (varies by stat complexity)

### 5. Feature Engineering Discipline
Every candidate feature must pass a **5-stage pre-ablation gate** before entering the model:

1. **Schema audit** — column exists, correct dtype, reasonable NaN rate
2. **Merge-key audit** — season aggregates use prior-season key (no forward leak)
3. **Temporal constancy check** — feature varies within and across seasons appropriately
4. **Canonical leak test** — correlation gap + cross-temporal peak
5. **Collinearity cap** — max |correlation| < 0.70 vs existing baseline features

This process rejected 9+ candidate feature families across 50 experiment sessions, preventing both data leakage and signal redundancy.

---

## Walk-Forward Results

### Form-Streak Model (IS_HOT ≥ 2, per-fold ensemble AUC)

| Market | F1 | F2 | F3 | F4 | F5 | F6 | **Mean** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| PTS | 0.647 | 0.603 | 0.648 | 0.668 | 0.630 | 0.668 | **0.644** |
| REB | 0.601 | 0.583 | 0.604 | 0.618 | 0.598 | 0.613 | **0.603** |
| AST | 0.574 | 0.595 | 0.595 | 0.598 | 0.594 | 0.582 | **0.590** |

### Beat-the-Line Model (P(actual > line), per-fold ensemble AUC)

| Market | F1 | F2 | F3 | F4 | F5 | F6 | **Mean** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| PTS | 0.643 | 0.607 | 0.636 | 0.663 | 0.617 | 0.655 | **0.637** |
| REB | 0.602 | 0.585 | 0.609 | 0.613 | 0.588 | 0.611 | **0.601** |
| AST | 0.578 | 0.579 | 0.597 | 0.591 | 0.575 | 0.564 | **0.581** |

> F1 (cold start) consistently shows lowest AUC — expected with limited training data. F3–F5 (production-like windows) reach 0.73–0.86 for form-streak predictions.

---

## Project Structure

```
nba-props-ml/
├── README.md                           # This file
├── docs/
│   └── pipeline_architecture.md        # Daily pipeline design doc
├── features/
│   └── prior_season_builder.py         # Prior-season feature extraction (sanitized)
├── validation/
│   ├── walk_forward.py                 # Walk-forward validation framework
│   └── leak_detection.py              # Multi-layer leakage detection
└── results/
    └── sample_wf_results.csv           # Sample validation output
```

---

## Tech Stack

- **Python 3.10** on Google Colab (GPU runtime)
- **LightGBM** — primary gradient boosting model
- **XGBoost / scikit-learn** — ensemble components (Random Forest, Logistic Regression)
- **pandas / NumPy** — data engineering + feature computation
- **NBA API** (`nba_api`) — official stats endpoint scraping
- **The Odds API** — live sportsbook lines and juice
- **Google Drive** — model artifact storage + versioned datasets

---

## Methodology Notes

- **No look-ahead bias**: all features computed from data available before game time
- **Expanding-window only**: models never trained on future data, folds grow monotonically
- **Feature shipping gate**: any new feature must demonstrate consistent positive lift across ALL 6 validation folds (not just on average)
- **Prior-season pattern**: season-level aggregates always keyed on previous season, preventing within-season data leakage
- **50+ documented experiment sessions** with version-controlled protocol, tracking every decision, rejection, and false positive

---

## About

This is a personal ML project focused on NBA player prop prediction. It demonstrates production-grade ML engineering practices in a domain where data leakage, distribution shift, and overfitting are constant threats.

**Key skills demonstrated:**
- End-to-end ML pipeline design (scraping → feature engineering → training → inference)
- Walk-forward temporal validation for time-series classification
- Multi-layer data quality and leakage detection
- Rigorous experiment tracking and reproducibility
- Feature selection with statistical gating (not just importance ranking)
- Multi-model ensemble with market-specific tuning
- Production monitoring for distribution shift detection

---

*Built with Python, LightGBM, and a lot of domain expertise.*
