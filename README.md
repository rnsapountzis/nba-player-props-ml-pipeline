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

## 📊 Synthetic Line Validation

Synthetic lines were generated from private pre-game player/team features.  
The public repository only includes validation code and summary results.  
On overlapping historical samples, synthetic estimates showed strong agreement with real sportsbook lines.

| Market | MAE | Within ±0.5 | Within ±1.0 | Correlation |
|--------|-----|------------|------------|-------------|
| PTS | 0.69 | 64.6% | 85.5% | 0.987 |
| REB | 0.26 | 94.5% | 99.1% | 0.985 |
| AST | 0.19 | 98.0% | 99.5% | 0.985 |

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

The system was evaluated under multiple target formulations using 6-fold expanding-window validation.

### OVER_LINE (Beat-the-Line Target)

| Market | Mean AUC |
|--------|----------|
| PTS | 0.697 |
| REB | 0.666 |
| AST | 0.681 |

### HOT_GE2 (Form / Streak Target)

| Market | Mean AUC |
|--------|----------|
| PTS | 0.794 |
| REB | 0.777 |
| AST | 0.792 |

### Interpretation

- OVER_LINE models capture direct line-beating behavior and provide stable performance across all markets  
- HOT_GE2 models achieve higher AUC due to modeling a more structured target (player form / streak dynamics)  
- The two targets capture different aspects of player performance and are used as complementary signals  


--------|-----|-----|-----|-----|-----|-----|
| PTS | 0.580 | 0.636 | 0.756 | 0.754 | 0.755 | 0.636 |
| REB | 0.607 | 0.670 | 0.732 | 0.741 | 0.731 | 0.633 |
| AST | 0.615 | 0.687 | 0.757 | 0.754 | 0.746 | 0.643 |

### Model B — HIT_OVER Classifier

Target: binary prediction of whether the player beats the prop line.

| Market | F1 | F2 | F3 | F4 | F5 | F6 |
|--------|-----|-----|-----|-----|-----|-----|
| PTS | 0.580 | 0.636 | 0.756 | 0.754 | 0.755 | 0.636 |
| REB | 0.607 | 0.670 | 0.732 | 0.741 | 0.731 | 0.633 |
| AST | 0.615 | 0.687 | 0.757 | 0.754 | 0.746 | 0.643 |

### Model C — IS_HOT ≥ 2 Form Model

Target: player short-term form/streak classification.

| Market | F1 | F2 | F3 | F4 | F5 | F6 |
|--------|-----|-----|-----|-----|-----|-----|
| PTS | 0.657 | 0.558 | 0.855 | 0.865 | 0.851 | 0.749 |
| REB | 0.670 | 0.566 | 0.847 | 0.856 | 0.857 | 0.738 |
| AST | 0.658 | 0.542 | 0.850 | 0.862 | 0.861 | 0.696 |

### Interpretation

- Model A and Model B show very similar fold-level behavior, suggesting both formulations capture comparable predictive signal.
- Model C produces higher AUC because it solves a different target: player form/streak detection rather than direct line prediction.
- F3–F5 represent the most stable production-like validation windows.
- F1/F2 are colder-start windows with less historical training context.

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

## 🔒 Code Scope

This repository focuses on:
- validation methodology
- walk-forward evaluation
- synthetic vs real line consistency checks

Core model training and feature construction layers are excluded, while preserving all components necessary to verify the system’s statistical behavior.

This separation reflects real-world ML system design, where validation and monitoring layers are decoupled from proprietary modeling logic.

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
