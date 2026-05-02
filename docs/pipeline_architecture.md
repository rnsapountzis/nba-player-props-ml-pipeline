# Daily Production Pipeline — Architecture

## Overview

The prediction pipeline runs daily during the NBA regular season (~October through April). Each run processes the previous night's games and generates predictions for upcoming games.

## Pipeline Stages

### Stage 1: Data Ingestion (`~5 min`)

**NBA API Scraper**
- Pulls player box scores via `nba_api.stats.endpoints.playergamelogs`
- Pulls team-level stats via `teamgamelogs`
- Filters by minimum minutes threshold (configurable, default ≥ 5 MIN)
- Auto-appends to season-long master CSV on Google Drive
- Handles API rate limiting (1-2s delay between calls)

**Tracking Data**
- Player tracking stats (touches, drives, catch-and-shoot, paint/elbow/post touches)
- Sourced from NBA.com advanced tracking endpoints
- Season-level aggregates stored as Excel files per category

**Odds Scraper**
- Pulls live player props from The Odds API (DraftKings primary)
- Markets: `player_points`, `player_rebounds`, `player_assists`
- Extracts over/under lines + juice for each player
- Falls back to synthetic lines from historical data when live props unavailable

### Stage 2: Data Alignment (`~2 min`)

**Player Name Normalization**
- Unicode normalization (NFD → ASCII, strip accents)
- Suffix removal (Jr, Sr, II, III, IV)
- Case-insensitive canonical form
- Handles edge cases: Nikola Jokić → NIKOLA JOKIC, Shai Gilgeous-Alexander → SHAI GILGEOUS-ALEXANDER

**Team + Season Merge**
- Game-level join: player stats ↔ team stats ↔ opponent stats
- Season derivation from game date (Oct-Dec → year X, Jan-Apr → year X-1)
- Historical props merge with fuzzy name matching (difflib, ≥ 0.85 threshold)

### Stage 3: Feature Engineering (`~10 min`)

**Rolling Window Features**
- Per-player rolling means/stds over last N games (N = 3, 5, 10, 15)
- Form streaks (consecutive games above/below thresholds)
- Recent deviation from season average

**Cross-Domain Interactions**
- Player form × opponent defensive tendencies
- Usage rate × game pace interactions
- Line deviation features (player average vs. sportsbook line)

**Prior-Season Aggregates**
- Season-level stats from 15+ NBA API endpoint groups
- Merged on PRIOR SEASON key (a player's 2023-24 stats inform 2024-25 predictions)
- 302 features from offensive efficiency, defensive tracking, playmaking, hustle, clutch, etc.

**Contextual Features**
- Home/away indicator
- Rest days since last game
- Game scheduling context (back-to-backs, national TV, etc.)

### Stage 4: Model Inference (`~3 min`)

**Multi-Model Ensemble**
- LightGBM (primary, highest individual AUC)
- XGBoost (secondary gradient booster)
- Random Forest (diversity via bagging)
- Logistic Regression (calibrated baseline)
- Weighted average ensemble with per-market optimized weights

**Meta-Learner**
- Stacked model that takes base model outputs as features
- Trained on out-of-fold predictions from walk-forward validation
- Per-market routing: enabled/disabled based on bootstrap CI analysis

**Market-Specific Feature Sets**
- Each market (PTS/REB/AST) uses its own curated feature set
- Feature counts: 67–404 per market (AST uses most due to playmaking complexity)
- Feature selection via walk-forward ablation with strict shipping gates

### Stage 5: Signal Generation (`~1 min`)

**Edge Detection**
- Multi-tier confidence buckets (STRONG, HOT, MODERATE, LOW)
- Model probability × line deviation × recent form = composite signal

**Output**
- Per-player, per-market probability predictions
- Confidence tier assignment
- Side recommendation (OVER/UNDER) with edge magnitude

## Data Flow

```
NBA API ──────┐
              ├──▶ Raw CSVs ──▶ Merged DF ──▶ Feature Matrix ──▶ Predictions
Odds API ─────┘    (Drive)      (in-memory)    (~900 cols)        (per-player)
Tracking Data ┘

Prior-Season ──▶ PRIORS_PLAYER_SEASON.csv ──▶ Left join on PREV_SEASON
(14 source groups, 302 features)
```

## Key Design Decisions

1. **Colab + Drive** for portability and GPU access without infrastructure cost
2. **CSV-based storage** for simplicity and human readability (no database needed at this scale)
3. **Per-market models** rather than a single multi-target model — different stats have different signal profiles
4. **Expanding-window only** — models always trained on all available history, never a sliding window
5. **LightGBM native NaN handling** — no imputation, no fill values, tree-based models handle missing data naturally
