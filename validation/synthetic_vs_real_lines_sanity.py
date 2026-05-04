"""
Synthetic vs Real Sportsbook Line Sanity Check
=============================================

This script validates precomputed synthetic NBA player prop lines against
real historical sportsbook lines.

Important:
- The synthetic line generation logic is intentionally NOT included.
- Private feature engineering, model training, and betting/edge logic are omitted.
- This script only evaluates already-generated synthetic lines.

Expected columns after merge:
    PLAYER
    GAME_DATE
    SYNTH_LINE_PTS, SYNTH_LINE_REB, SYNTH_LINE_AST
    PROP_LINE_PTS,  PROP_LINE_REB,  PROP_LINE_AST

Optional role/tier column:
    SLOT_FGA
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MARKETS = ["PTS", "REB", "AST"]


def normalize_player_name(s: pd.Series) -> pd.Series:
    """Light normalization for safer player-name matching."""
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV and standardize basic merge columns."""
    df = pd.read_csv(path)

    if "GAME_DATE" not in df.columns:
        raise ValueError(f"{path} is missing GAME_DATE")

    if "PLAYER" not in df.columns:
        raise ValueError(f"{path} is missing PLAYER")

    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.date
    df["PLAYER_KEY"] = normalize_player_name(df["PLAYER"])

    return df


def merge_synthetic_and_real(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge synthetic-line data with real sportsbook-line data."""
    real_cols = ["PLAYER_KEY", "GAME_DATE"]

    for market in MARKETS:
        col = f"PROP_LINE_{market}"
        if col in real_df.columns:
            real_cols.append(col)

    missing_real = [f"PROP_LINE_{m}" for m in MARKETS if f"PROP_LINE_{m}" not in real_df.columns]
    if missing_real:
        raise ValueError(f"Real-line dataset missing columns: {missing_real}")

    synth_missing = [f"SYNTH_LINE_{m}" for m in MARKETS if f"SYNTH_LINE_{m}" not in synthetic_df.columns]
    if synth_missing:
        raise ValueError(f"Synthetic-line dataset missing columns: {synth_missing}")

    # Keep one sportsbook line row per player/date if duplicates exist.
    real_small = (
        real_df[real_cols]
        .dropna(subset=["PLAYER_KEY", "GAME_DATE"])
        .drop_duplicates(["PLAYER_KEY", "GAME_DATE"])
    )

    merged = synthetic_df.merge(
        real_small,
        on=["PLAYER_KEY", "GAME_DATE"],
        how="inner",
        suffixes=("", "_REAL"),
    )

    return merged


def compute_market_metrics(df: pd.DataFrame, markets: Iterable[str] = MARKETS) -> pd.DataFrame:
    """Compute clean SYNTH_LINE vs PROP_LINE validation metrics."""
    rows = []

    for market in markets:
        synth_col = f"SYNTH_LINE_{market}"
        real_col = f"PROP_LINE_{market}"

        sub = df[[synth_col, real_col]].copy()
        sub[synth_col] = pd.to_numeric(sub[synth_col], errors="coerce")
        sub[real_col] = pd.to_numeric(sub[real_col], errors="coerce")
        sub = sub.dropna()

        if sub.empty:
            rows.append({
                "market": market,
                "n_overlap": 0,
                "mae": np.nan,
                "mean_diff": np.nan,
                "exact_match": np.nan,
                "within_0.5": np.nan,
                "within_1.0": np.nan,
                "correlation": np.nan,
            })
            continue

        diff = sub[synth_col] - sub[real_col]
        abs_diff = diff.abs()

        rows.append({
            "market": market,
            "n_overlap": int(len(sub)),
            "mae": round(float(abs_diff.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "exact_match": round(float((abs_diff < 1e-9).mean()), 4),
            "within_0.5": round(float((abs_diff <= 0.5).mean()), 4),
            "within_1.0": round(float((abs_diff <= 1.0).mean()), 4),
            "correlation": round(float(sub[synth_col].corr(sub[real_col])), 4),
        })

    return pd.DataFrame(rows)


def compute_role_tier_metrics(df: pd.DataFrame, role_col: str = "SLOT_FGA") -> pd.DataFrame:
    """
    Optional role/tier sanity check.

    This checks whether synthetic-line error is stable across player tiers.
    The role/tier generation logic is not included here.
    """
    if role_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for market in MARKETS:
        synth_col = f"SYNTH_LINE_{market}"
        real_col = f"PROP_LINE_{market}"

        sub = df[[role_col, synth_col, real_col]].copy()
        sub[synth_col] = pd.to_numeric(sub[synth_col], errors="coerce")
        sub[real_col] = pd.to_numeric(sub[real_col], errors="coerce")
        sub = sub.dropna()

        if sub.empty:
            continue

        sub["abs_diff"] = (sub[synth_col] - sub[real_col]).abs()

        grouped = (
            sub.groupby(role_col)["abs_diff"]
            .agg(["count", "mean", "median", "max"])
            .reset_index()
        )

        for _, r in grouped.iterrows():
            rows.append({
                "market": market,
                "role_tier": r[role_col],
                "count": int(r["count"]),
                "mae": round(float(r["mean"]), 4),
                "median_abs_error": round(float(r["median"]), 4),
                "max_abs_error": round(float(r["max"]), 4),
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate precomputed synthetic NBA prop lines against real sportsbook lines."
    )
    parser.add_argument("--synthetic-path", required=True, help="CSV with SYNTH_LINE_* columns.")
    parser.add_argument("--real-path", required=True, help="CSV with PROP_LINE_* columns.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where validation CSV outputs will be saved.",
    )
    parser.add_argument(
        "--role-col",
        default="SLOT_FGA",
        help="Optional player-tier column for role/tier consistency checks.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_df = load_dataset(args.synthetic_path)
    real_df = load_dataset(args.real_path)

    merged = merge_synthetic_and_real(synthetic_df, real_df)

    summary = compute_market_metrics(merged)
    summary_path = output_dir / "synthetic_vs_real_summary.csv"
    summary.to_csv(summary_path, index=False)

    role_summary = compute_role_tier_metrics(merged, role_col=args.role_col)
    if not role_summary.empty:
        role_path = output_dir / "synthetic_vs_real_by_role_tier.csv"
        role_summary.to_csv(role_path, index=False)

    print("\nSynthetic vs Real Line Summary")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    if not role_summary.empty:
        print(f"Saved: {role_path}")


if __name__ == "__main__":
    main()
