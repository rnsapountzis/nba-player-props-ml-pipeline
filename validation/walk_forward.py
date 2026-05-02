"""
Walk-Forward Validation Framework for NBA Props
=================================================
Date-based expanding window validation with multi-model ensemble.

Key design constraints:
  - NEVER random splits — games are time-ordered
  - Expanding window: each fold trains on ALL prior data
  - Fold edges computed from date percentiles, not row quantiles
  - 6 folds primary (0.30 → 1.00), 13 folds for robustness checks

This file is a sanitized excerpt — feature names and model paths removed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from collections import defaultdict


# ============================================================
# WALK-FORWARD FOLD CONSTRUCTION
# ============================================================

def build_wf_folds(df, date_col='GAME_DATE', n_folds=6, start_pct=0.30):
    """
    Build date-based expanding window folds.
    
    Logic:
      - Sort unique dates
      - Compute fold edges at percentile positions: [0.30, 0.4167, ..., 1.00]
      - Fold i trains on dates[0 : edge[i]], tests on dates[edge[i] : edge[i+1]]
      - Each subsequent fold's training set includes ALL prior test data
    
    Parameters
    ----------
    df : DataFrame with a date column
    date_col : name of the date column
    n_folds : number of validation folds (default 6)
    start_pct : fraction of dates used as minimum training set (default 0.30)
    
    Returns
    -------
    list of (train_mask, test_mask) boolean arrays
    """
    dates_sorted = np.sort(df[date_col].unique())
    n_dates = len(dates_sorted)
    
    # Percentile-based edges — NOT row-quantile
    edges_pct = np.linspace(start_pct, 1.00, n_folds + 1)
    edge_idx = [min(int(e * n_dates), n_dates - 1) for e in edges_pct]
    edge_dates = [dates_sorted[i] for i in edge_idx]
    
    folds = []
    for i in range(n_folds):
        train_end = edge_dates[i]
        test_start = edge_dates[i]
        test_end = edge_dates[i + 1]
        
        train_mask = df[date_col] < train_end
        test_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)
        
        # Last fold includes the boundary
        if i == n_folds - 1:
            test_mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)
        
        folds.append((train_mask, test_mask))
    
    return folds


# ============================================================
# SINGLE-MODEL WALK-FORWARD TRAINING
# ============================================================

def train_wf_lgb(df, features, target, folds, lgb_params=None):
    """
    Train LightGBM across walk-forward folds, collecting OOF predictions.
    
    Parameters
    ----------
    df : DataFrame
    features : list of feature column names
    target : target column name (binary)
    folds : list of (train_mask, test_mask) from build_wf_folds
    lgb_params : dict of LightGBM parameters (sensible defaults provided)
    
    Returns
    -------
    oof_preds : array of OOF predictions (NaN where not in any test fold)
    fold_aucs : list of per-fold AUC scores
    models : list of trained Booster objects
    """
    if lgb_params is None:
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.03,
            'num_leaves': 63,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'seed': 42,
        }
    
    oof_preds = np.full(len(df), np.nan)
    fold_aucs = []
    models = []
    
    for i, (train_mask, test_mask) in enumerate(folds):
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, target]
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, target]
        
        # Drop rows with NaN target
        valid_train = y_train.notna()
        valid_test = y_test.notna()
        
        dtrain = lgb.Dataset(
            X_train[valid_train], 
            label=y_train[valid_train]
        )
        dval = lgb.Dataset(
            X_test[valid_test], 
            label=y_test[valid_test], 
            reference=dtrain
        )
        
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        
        preds = model.predict(X_test[valid_test])
        oof_preds[test_mask.values & valid_test.values] = preds
        
        auc = roc_auc_score(y_test[valid_test], preds)
        fold_aucs.append(auc)
        models.append(model)
        
        print(f"  Fold {i+1}: AUC={auc:.4f} | "
              f"train={valid_train.sum():,} | test={valid_test.sum():,}")
    
    return oof_preds, fold_aucs, models


# ============================================================
# MULTI-MODEL ENSEMBLE
# ============================================================

def train_wf_ensemble(df, features, target, folds, 
                       models_to_use=('lgb', 'xgb', 'rf', 'lr')):
    """
    Train multiple model types across walk-forward folds and
    compute a weighted ensemble.
    
    Returns per-model OOF predictions and ensemble OOF.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    all_oof = {}
    all_aucs = {}
    
    for model_type in models_to_use:
        print(f"\n{'='*50}")
        print(f"Training: {model_type.upper()}")
        print('='*50)
        
        if model_type == 'lgb':
            oof, aucs, _ = train_wf_lgb(df, features, target, folds)
        elif model_type == 'xgb':
            oof, aucs = _train_wf_xgb(df, features, target, folds)
        elif model_type == 'rf':
            oof, aucs = _train_wf_sklearn(
                df, features, target, folds,
                RandomForestClassifier(n_estimators=300, max_depth=12, 
                                       min_samples_leaf=30, random_state=42,
                                       n_jobs=-1)
            )
        elif model_type == 'lr':
            oof, aucs = _train_wf_sklearn(
                df, features, target, folds,
                LogisticRegression(max_iter=1000, C=0.1, random_state=42),
                scale=True
            )
        
        all_oof[model_type] = oof
        all_aucs[model_type] = aucs
    
    # Weighted ensemble (equal weights as default)
    valid_mask = np.all([~np.isnan(oof) for oof in all_oof.values()], axis=0)
    oof_matrix = np.column_stack([all_oof[m] for m in models_to_use])
    
    # Simple average ensemble
    ens_oof = np.full(len(df), np.nan)
    ens_oof[valid_mask] = oof_matrix[valid_mask].mean(axis=1)
    
    # Compute per-fold ensemble AUC
    target_vals = df[target].values
    ens_aucs = []
    for _, test_mask in folds:
        mask = test_mask.values & valid_mask & ~np.isnan(target_vals)
        if mask.sum() > 0:
            auc = roc_auc_score(target_vals[mask], ens_oof[mask])
            ens_aucs.append(auc)
    
    all_oof['ensemble'] = ens_oof
    all_aucs['ensemble'] = ens_aucs
    
    return all_oof, all_aucs


def _train_wf_sklearn(df, features, target, folds, estimator, scale=False):
    """Generic sklearn model walk-forward training."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    
    oof = np.full(len(df), np.nan)
    aucs = []
    
    for i, (train_mask, test_mask) in enumerate(folds):
        X_tr = df.loc[train_mask, features].values
        y_tr = df.loc[train_mask, target].values
        X_te = df.loc[test_mask, features].values
        y_te = df.loc[test_mask, target].values
        
        valid_tr = ~np.isnan(y_tr)
        valid_te = ~np.isnan(y_te)
        
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
        X_te_v, y_te_v = X_te[valid_te], y_te[valid_te]
        
        # Impute NaN for sklearn models
        imp = SimpleImputer(strategy='median')
        X_tr = imp.fit_transform(X_tr)
        X_te_v = imp.transform(X_te_v)
        
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te_v = scaler.transform(X_te_v)
        
        model = clone(estimator)
        model.fit(X_tr, y_tr)
        
        preds = model.predict_proba(X_te_v)[:, 1]
        oof[test_mask.values & valid_te] = preds  # simplified indexing
        
        auc = roc_auc_score(y_te_v, preds)
        aucs.append(auc)
        print(f"  Fold {i+1}: AUC={auc:.4f}")
    
    return oof, aucs


def _train_wf_xgb(df, features, target, folds):
    """XGBoost walk-forward training."""
    import xgboost as xgb
    
    oof = np.full(len(df), np.nan)
    aucs = []
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 50,
        'seed': 42,
        'verbosity': 0,
    }
    
    for i, (train_mask, test_mask) in enumerate(folds):
        X_tr = df.loc[train_mask, features]
        y_tr = df.loc[train_mask, target]
        X_te = df.loc[test_mask, features]
        y_te = df.loc[test_mask, target]
        
        valid_tr = y_tr.notna()
        valid_te = y_te.notna()
        
        dtrain = xgb.DMatrix(X_tr[valid_tr], label=y_tr[valid_tr])
        dval = xgb.DMatrix(X_te[valid_te], label=y_te[valid_te])
        
        model = xgb.train(
            params, dtrain, 2000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        oof[test_mask.values & valid_te.values] = preds
        
        auc = roc_auc_score(y_te[valid_te], preds)
        aucs.append(auc)
        print(f"  Fold {i+1}: AUC={auc:.4f}")
    
    return oof, aucs


# ============================================================
# ABLATION GATE — Feature Shipping Decision
# ============================================================

def evaluate_ablation(base_aucs, candidate_aucs, 
                       path='B', min_delta=0.0015):
    """
    Evaluate whether a candidate feature set passes the shipping gate.
    
    Gate paths:
      Path A (strict): ΔAUC ≥ +0.0020, ≥ 75% positive folds, base ≥ 0.60
      Path B (default): ΔAUC ≥ +0.0015, 100% positive folds (6/6)
      Path C (orthogonal): ΔAUC ≥ +0.0010, 100% positive folds
    
    Hard reject: any fold with ΔAUC < -0.005
    
    Returns
    -------
    dict with 'verdict', 'delta_mean', 'positive_folds', 'per_fold_delta'
    """
    base = np.array(base_aucs)
    cand = np.array(candidate_aucs)
    deltas = cand - base
    
    delta_mean = deltas.mean()
    pos_folds = (deltas > 0).sum()
    n_folds = len(deltas)
    worst_fold = deltas.min()
    
    # Hard reject
    if worst_fold < -0.005:
        verdict = 'REJECT (catastrophic fold)'
    elif path == 'A':
        if delta_mean >= 0.0020 and pos_folds >= 0.75 * n_folds and base.mean() >= 0.60:
            verdict = 'PASS (Path A)'
        else:
            verdict = 'REJECT'
    elif path == 'B':
        if delta_mean >= min_delta and pos_folds == n_folds:
            verdict = 'PASS (Path B)'
        else:
            verdict = 'REJECT'
    elif path == 'C':
        if delta_mean >= 0.0010 and pos_folds == n_folds:
            verdict = 'PASS (Path C)'
        else:
            verdict = 'REJECT'
    else:
        verdict = 'UNKNOWN PATH'
    
    return {
        'verdict': verdict,
        'delta_mean': round(delta_mean, 5),
        'positive_folds': f"{pos_folds}/{n_folds}",
        'worst_fold_delta': round(worst_fold, 5),
        'per_fold_delta': [round(d, 5) for d in deltas],
    }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == '__main__':
    """
    Example usage (pseudocode — actual feature names removed):
    
    df = pd.read_csv('dataset.csv', low_memory=False)
    
    # Build target
    df['TARGET'] = (df['FORM_METRIC'] >= 2).astype(int)
    
    # Build folds
    folds = build_wf_folds(df, date_col='GAME_DATE', n_folds=6)
    
    # Train baseline
    baseline_features = load_feature_list('baseline.pkl')
    _, base_aucs, _ = train_wf_lgb(df, baseline_features, 'TARGET', folds)
    
    # Train candidate (baseline + new features)
    candidate_features = baseline_features + new_features
    _, cand_aucs, _ = train_wf_lgb(df, candidate_features, 'TARGET', folds)
    
    # Gate decision
    result = evaluate_ablation(base_aucs, cand_aucs, path='B')
    print(result)
    # {'verdict': 'PASS (Path B)', 'delta_mean': 0.0018, 
    #  'positive_folds': '6/6', ...}
    """
    print("Walk-forward validation framework loaded.")
    print("See docstrings for usage.")
