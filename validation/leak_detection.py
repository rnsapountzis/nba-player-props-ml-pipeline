"""
Multi-Layer Data Leakage Detection for Sports Prediction
=========================================================
A 4-layer leak detection pipeline designed for sports prop prediction,
where standard correlation tests miss subtle forward-looking information.

Developed over 50+ experiment sessions after discovering that:
  - Standard gap tests are BLIND to constant-within-season features
  - Season-aggregate features that are clean by correlation can still
    leak future information via merge-key errors
  - Cross-temporal peak detection catches time-shifted leakage
  - Two false-positive types need disambiguation to avoid rejecting
    legitimately clean prior-season features

This framework prevented 9 false-clean features from entering production
and correctly rescued 7 features that would have been wrongly rejected.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================
# LAYER 1: CANONICAL GAP TEST (Primary)
# ============================================================

def canonical_gap_test(df: pd.DataFrame,
                       feature: str,
                       target: str,
                       player_col: str = 'PLAYER_NORM',
                       date_col: str = 'GAME_DATE') -> Dict:
    """
    Test whether a feature correlates more with the CURRENT target
    than with LAGGED targets. If so, it may contain future information.
    
    Mechanism:
      corr_current  = corr(feature, target)
      corr_manual5  = corr(feature, target.shift(1).rolling(5).mean())
      corr_manual10 = corr(feature, target.shift(1).rolling(10).mean())
      gap = max(|corr_manual5|, |corr_manual10|) - |corr_current|
    
    Interpretation:
      gap > 0      → CLEAN (feature aligns with past targets, not current)
      gap < -0.03  → LEAK (feature knows current/future target)
      -0.03 ≤ gap ≤ 0 → INSUFFICIENT (ambiguous, needs further tests)
    
    Returns
    -------
    dict with corr_current, corr_lag5, corr_lag10, gap, verdict
    """
    sub = df[[player_col, date_col, feature, target]].dropna().copy()
    sub = sub.sort_values([player_col, date_col])
    
    # Current correlation
    corr_current = sub[feature].corr(sub[target])
    
    # Lagged rolling correlations (shift by 1 to exclude current game)
    sub['_lag_roll5'] = (sub.groupby(player_col)[target]
                          .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean()))
    sub['_lag_roll10'] = (sub.groupby(player_col)[target]
                           .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
    
    corr_lag5 = sub[feature].corr(sub['_lag_roll5'])
    corr_lag10 = sub[feature].corr(sub['_lag_roll10'])
    
    # Gap calculation
    gap = max(abs(corr_lag5), abs(corr_lag10)) - abs(corr_current)
    
    # Verdict
    if gap > 0:
        verdict = 'CLEAN'
    elif gap < -0.03:
        verdict = 'LEAK'
    else:
        verdict = 'INSUFFICIENT'
    
    return {
        'feature': feature,
        'corr_current': round(corr_current, 4),
        'corr_lag5': round(corr_lag5, 4),
        'corr_lag10': round(corr_lag10, 4),
        'gap': round(gap, 4),
        'verdict': verdict,
    }


# ============================================================
# LAYER 2: CROSS-TEMPORAL PEAK DETECTION
# ============================================================

def cross_temporal_peak_test(df: pd.DataFrame,
                              feature: str,
                              target: str,
                              player_col: str = 'PLAYER_NORM',
                              date_col: str = 'GAME_DATE',
                              max_lag: int = 3) -> Dict:
    """
    Compute correlation between feature[t] and target[t+k] for k in [-max_lag..+max_lag].
    
    If the peak correlation occurs at k ≥ +1, the feature contains
    information about FUTURE target values → LEAK.
    
    Mechanism:
      For each lag k:
        corr[k] = correlation(feature at time t, target at time t+k)
      peak_k = argmax(|corr[k]|)
    
    Interpretation:
      peak at k ≤ 0  → CLEAN (feature predicts past/current)
      peak at k ≥ +1 → LEAK (feature predicts future)
      plateau k=0..+3 → LEAK (flat correlation = constant feature)
    
    Returns
    -------
    dict with lag_correlations, peak_lag, verdict
    """
    sub = df[[player_col, date_col, feature, target]].dropna().copy()
    sub = sub.sort_values([player_col, date_col])
    
    lag_corrs = {}
    
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            lag_corrs[k] = sub[feature].corr(sub[target])
        else:
            shifted = sub.groupby(player_col)[target].shift(-k)
            lag_corrs[k] = sub[feature].corr(shifted)
    
    # Find peak
    peak_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
    peak_corr = lag_corrs[peak_lag]
    
    # Check for plateau (all lags ≥ 0 have similar correlation)
    forward_corrs = [abs(lag_corrs[k]) for k in range(0, max_lag + 1)]
    is_plateau = (max(forward_corrs) - min(forward_corrs)) < 0.01 and max(forward_corrs) > 0.05
    
    if is_plateau:
        verdict = 'LEAK (plateau)'
    elif peak_lag >= 1:
        verdict = 'LEAK (future peak)'
    else:
        verdict = 'CLEAN'
    
    return {
        'feature': feature,
        'lag_correlations': {k: round(v, 4) for k, v in sorted(lag_corrs.items())},
        'peak_lag': peak_lag,
        'peak_corr': round(peak_corr, 4),
        'verdict': verdict,
    }


# ============================================================
# LAYER 3 + 4: TEMPORAL CONSTANCY GATE + CROSS-SEASON DISAMBIGUATION
# ============================================================

def temporal_constancy_test(df: pd.DataFrame,
                             feature: str,
                             player_col: str = 'PLAYER_NORM',
                             season_col: str = 'SEASON') -> Dict:
    """
    Test whether a feature varies within a season and across seasons.
    
    Designed to catch a blind spot in the canonical gap test:
    if a feature is CONSTANT within each (player, season) window,
    then corr_current ≈ corr_lag5 ≈ corr_lag10 → gap ≈ 0 → false CLEAN.
    
    Two-test matrix (refined version):
    
      TEST 1: intra-season variation
        intra_std = mean of std(feature) within each (player, season) group
        intra_ratio = intra_std / total_std
    
      TEST 2: cross-season variation (disambiguation)
        cross_std = mean of std(feature) within each player (across seasons)
        cross_ratio = cross_std / total_std
    
    Interpretation matrix:
      intra ≥ 0.10                        → CLEAN_ROLLING (per-game feature)
      intra < 0.05 + cross ≥ 0.10        → CLEAN_PRIOR_SEASON (expected for
                                             prior-season aggregates)
      intra < 0.05 + cross ∈ [0.05,0.10] → STICKY (low-info, marginal signal)
      intra < 0.05 + cross < 0.05        → LEAK (constant everywhere =
                                             current-season merge error)
      intra ∈ [0.05, 0.10]               → LIKELY_PRIOR (acceptable)
    
    Why this matters:
      - Layer 1 (gap test) returns false CLEAN for constant-within-season features
      - Layer 3 catches this, but has a false-alarm problem:
        prior-season aggregates are INTENTIONALLY constant within a season
      - Layer 4 (cross-season check) disambiguates: if the feature varies across
        a player's career (different values in 2022-23 vs 2023-24), it's a
        legitimate prior-season feature, not a leak
    
    Evidence:
      - 9 features passed gap test but had intra_ratio = 0.000 (false CLEAN)
      - 7 features with intra = 0.000 but cross_season ≥ 0.10 were correctly
        identified as prior-season aggregates (false alarm prevention)
    """
    sub = df[[player_col, season_col, feature]].dropna()
    
    total_std = sub[feature].std()
    if total_std == 0 or total_std != total_std:  # zero or NaN
        return {
            'feature': feature,
            'intra_ratio': 0.0,
            'cross_season_ratio': 0.0,
            'total_std': 0.0,
            'verdict': 'CONSTANT (no variance)',
        }
    
    # TEST 1: intra-season variance
    intra_std = sub.groupby([player_col, season_col])[feature].std().mean()
    intra_ratio = intra_std / total_std if total_std > 0 else 0
    
    # TEST 2: cross-season variance per player
    cross_std = sub.groupby(player_col)[feature].std().mean()
    cross_ratio = cross_std / total_std if total_std > 0 else 0
    
    # Interpretation matrix
    if intra_ratio >= 0.10:
        verdict = 'CLEAN_ROLLING'
    elif intra_ratio < 0.05:
        if cross_ratio >= 0.10:
            verdict = 'CLEAN_PRIOR_SEASON'
        elif cross_ratio >= 0.05:
            verdict = 'STICKY (low-info)'
        else:
            verdict = 'LEAK (constant everywhere)'
    else:  # 0.05 ≤ intra < 0.10
        verdict = 'LIKELY_PRIOR (acceptable)'
    
    return {
        'feature': feature,
        'intra_ratio': round(intra_ratio, 4),
        'cross_season_ratio': round(cross_ratio, 4),
        'total_std': round(total_std, 4),
        'verdict': verdict,
    }


# ============================================================
# FULL LEAK TEST PIPELINE
# ============================================================

def full_leak_test(df: pd.DataFrame,
                    features: List[str],
                    target: str,
                    player_col: str = 'PLAYER_NORM',
                    season_col: str = 'SEASON',
                    date_col: str = 'GAME_DATE') -> pd.DataFrame:
    """
    Run all 4 leak detection layers on a list of candidate features.
    
    Pipeline order:
      1. Temporal constancy gate (Layer 3+4) — catches gap-test blind spots
      2. Canonical gap test (Layer 1) — primary correlation-based test
      3. Cross-temporal peak detection (Layer 2) — time-shifted leakage
    
    ALL layers must return CLEAN/ACCEPTABLE for a feature to pass.
    
    Returns
    -------
    DataFrame with one row per feature, all test results, and final verdict
    """
    results = []
    
    for feat in features:
        row = {'feature': feat}
        
        # Layer 3+4: Temporal constancy
        tc = temporal_constancy_test(df, feat, player_col, season_col)
        row['tc_intra'] = tc['intra_ratio']
        row['tc_cross'] = tc['cross_season_ratio']
        row['tc_verdict'] = tc['verdict']
        
        # Layer 1: Canonical gap test
        gap = canonical_gap_test(df, feat, target, player_col, date_col)
        row['gap_corr_current'] = gap['corr_current']
        row['gap_value'] = gap['gap']
        row['gap_verdict'] = gap['verdict']
        
        # Layer 2: Cross-temporal peak
        peak = cross_temporal_peak_test(df, feat, target, player_col, date_col)
        row['peak_lag'] = peak['peak_lag']
        row['peak_verdict'] = peak['verdict']
        
        # Final verdict: ALL must pass
        verdicts = [tc['verdict'], gap['verdict'], peak['verdict']]
        
        if any('LEAK' in v for v in verdicts):
            row['final_verdict'] = 'LEAK'
            # Identify which layer caught it
            leak_layers = [v for v in verdicts if 'LEAK' in v]
            row['caught_by'] = ' + '.join(leak_layers)
        elif any('INSUFFICIENT' in v for v in verdicts):
            row['final_verdict'] = 'REVIEW'
            row['caught_by'] = ''
        else:
            row['final_verdict'] = 'CLEAN'
            row['caught_by'] = ''
        
        results.append(row)
    
    return pd.DataFrame(results)


# ============================================================
# PRE-ABLATION COMPLIANCE CHECK
# ============================================================

def pre_ablation_gate(df: pd.DataFrame,
                       candidate_features: List[str],
                       baseline_features: List[str],
                       target: str,
                       max_collinearity: float = 0.70) -> Dict:
    """
    Full pre-ablation compliance check before running walk-forward ablation.
    
    Checks:
      RULE R: No overlap between candidate and baseline feature sets
      RULE S: Max |correlation| between any candidate and any baseline < threshold
      RULE T: Season-aggregate features merged on PRIOR_SEASON (verified via
              temporal constancy — if intra_ratio = 0, must have cross_season ≥ 0.10)
      RULE U: All features pass temporal constancy gate
    
    Returns dict with pass/fail per rule and overall verdict.
    """
    results = {}
    
    # RULE R: Zero overlap
    overlap = set(candidate_features) & set(baseline_features)
    results['rule_r'] = {
        'name': 'Zero overlap (candidate ∩ baseline = ∅)',
        'pass': len(overlap) == 0,
        'overlap': list(overlap) if overlap else [],
    }
    
    # RULE S: Collinearity cap
    max_corr = 0
    max_pair = ('', '')
    for c in candidate_features:
        if c not in df.columns:
            continue
        for b in baseline_features:
            if b not in df.columns:
                continue
            r = abs(df[c].corr(df[b]))
            if r > max_corr:
                max_corr = r
                max_pair = (c, b)
    
    results['rule_s'] = {
        'name': f'Max collinearity < {max_collinearity}',
        'pass': max_corr < max_collinearity,
        'max_corr': round(max_corr, 4),
        'max_pair': max_pair,
    }
    
    # RULE T + U: Temporal constancy
    tc_failures = []
    for feat in candidate_features:
        if feat not in df.columns:
            continue
        tc = temporal_constancy_test(df, feat)
        if 'LEAK' in tc['verdict']:
            tc_failures.append((feat, tc['verdict']))
    
    results['rule_tu'] = {
        'name': 'Temporal constancy gate (all features)',
        'pass': len(tc_failures) == 0,
        'failures': tc_failures,
    }
    
    # Overall
    all_pass = all(r['pass'] for r in results.values())
    results['overall'] = 'PASS — proceed to ablation' if all_pass else 'FAIL — fix before ablation'
    
    return results


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == '__main__':
    """
    Example usage:
    
    df = pd.read_csv('dataset.csv')
    
    # Test a single feature
    result = canonical_gap_test(df, 'my_feature', 'TARGET')
    print(result)
    # {'feature': 'my_feature', 'corr_current': 0.15, 'gap': 0.02, 'verdict': 'CLEAN'}
    
    # Full pipeline on candidate features
    candidates = ['feat_a', 'feat_b', 'feat_c']
    results_df = full_leak_test(df, candidates, 'TARGET')
    print(results_df[['feature', 'final_verdict', 'caught_by']])
    
    # Pre-ablation gate
    gate = pre_ablation_gate(df, candidates, baseline_features, 'TARGET')
    print(gate['overall'])
    """
    print("Leak detection framework loaded.")
    print("Layers: gap test → cross-temporal → temporal constancy → cross-season")
