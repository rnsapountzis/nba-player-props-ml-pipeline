"""
Prior-Season Feature Builder
==============================
Extracts season-level player aggregates from multiple NBA data sources
and merges them as PRIOR-SEASON lookbacks for prediction models.

Key design principle:
  A player's 2023-24 season profile informs their 2024-25 game predictions.
  This prevents within-season data leakage while capturing stable player
  tendencies (offensive role, defensive style, playmaking profile).

Sources:
  1. Player tracking data (catch-and-shoot, drives, post/elbow/paint touches)
  2. League dashboard stats (usage, scoring, defense, misc)
  3. Hustle stats, clutch performance, point-of-attack defense
  4. Passing data (assist creation, pass volume)
  5. Possessions/touches data (ball-handling, time of possession)

Output:
  302 features across 14 source groups, covering 7 NBA seasons.
  
This file is a sanitized excerpt — specific column names generalized,
model paths removed. The methodology and architecture are production code.
"""

import os, re, glob, json, zipfile, unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, List, Tuple


# ============================================================
# NAME NORMALIZATION (production-locked)
# ============================================================

def canonical_norm(name: str) -> Optional[str]:
    """
    Normalize player names to a canonical form for cross-source matching.
    
    Handles:
      - Unicode characters (Jokić → JOKIC, Dončić → DONCIC)
      - Suffixes (Jr, Sr, II, III, IV)
      - Punctuation and apostrophes
      - Case normalization
    
    Examples:
      "Nikola Jokić"           → "NIKOLA JOKIC"
      "Jaren Jackson Jr."      → "JAREN JACKSON"
      "Shai Gilgeous-Alexander" → "SHAI GILGEOUS-ALEXANDER"
    """
    if pd.isna(name):
        return np.nan
    s = unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('ascii')
    s = s.upper().replace("'", "").replace(".", "").replace(",", "")
    s = re.sub(r'\s+(JR|SR|II|III|IV)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip() or np.nan


def prior_season(season: str) -> Optional[str]:
    """
    Map a season string to its prior season.
    '2024-25' → '2023-24'
    """
    m = re.match(r"(\d{4})-(\d{2})$", str(season))
    if not m:
        return None
    y = int(m.group(1))
    return f"{y-1}-{str(y % 100).zfill(2)}"


# ============================================================
# NBA API JSON PARSER
# ============================================================

def parse_nba_api_json(raw_data) -> pd.DataFrame:
    """
    Parse NBA API resultSets format into a DataFrame.
    
    The NBA stats API returns data in this structure:
      {"resultSets": [{"headers": ["COL1", "COL2", ...],
                        "rowSet": [[val1, val2, ...], ...]}]}
    
    Handles variations: resultsets, resultSet, direct headers+rowSet.
    Returns empty DataFrame on parse failure.
    """
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except (json.JSONDecodeError, TypeError):
        return pd.DataFrame()
    
    result_sets = (data.get('resultSets') or 
                   data.get('resultsets') or 
                   data.get('resultSet'))
    
    if not result_sets:
        if 'headers' in data and 'rowSet' in data:
            result_sets = [data]
        else:
            return pd.DataFrame()
    
    rs = result_sets[0] if isinstance(result_sets, list) else result_sets
    headers = rs.get('headers', [])
    rows = rs.get('rowSet', [])
    
    if not headers or not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=headers)


# ============================================================
# SOURCE COLLECTION
# ============================================================

class PriorSeasonBuilder:
    """
    Collects player statistics from multiple NBA data sources,
    aggregates them to season-level means, and produces a prior-season
    feature library for model training.
    
    Architecture:
      1. COLLECT: scan filesystem for data files (xlsx, csv, json-in-zip)
      2. PARSE: each source → (PLAYER_KEY, SEASON, feature_1, ..., feature_N)
      3. AGGREGATE: groupby(PLAYER_KEY, SEASON).mean() → one row per player-season
      4. COMBINE: merge all source groups via combine_first (outer join)
      5. MERGE: left join on PRIOR_SEASON into main game-level DataFrame
    
    The PRIOR_SEASON merge is critical: it prevents within-season data leakage.
    A player's 2023-24 stats are only used for 2024-25 predictions.
    """
    
    # Columns to exclude from feature engineering
    SKIP_PATTERN = re.compile(
        r'RANK|TEAM_ID|TEAM_ABBR|TEAM_COUNT|NICKNAME|PLAYER_ID|'
        r'PLAYER_NAME|PLAYER_LAST|GROUP_SET|PLAYER_POSITION|OBJECT',
        re.I
    )
    
    def __init__(self):
        self.sources = []  # (filepath, format, group, season)
        self.priors = None
    
    def add_excel_sources(self, directory: str, groups: Dict[str, str]):
        """
        Scan directory for Excel files with player tracking data.
        
        Parameters
        ----------
        directory : path to scan
        groups : dict mapping filename prefix to group name
                 e.g. {'CatchShoot': 'CATCH', 'Drives': 'DRIVE'}
        """
        for prefix, group in groups.items():
            pattern = os.path.join(directory, f"{prefix}_Player_*.xlsx")
            for f in glob.glob(pattern):
                m = re.search(r'(\d{4})(\d{2})\.xlsx$', f)
                if m:
                    season = f"{int(m.group(1))}-{m.group(2).zfill(2)}"
                    self.sources.append((f, 'xlsx', group, season))
    
    def add_csv_directory(self, directory: str, group: str, 
                          season_pattern: str = r'(\d{4}-\d{2})'):
        """Add all CSVs from a directory, parsing season from filename."""
        if not os.path.exists(directory):
            return
        for f in sorted(glob.glob(os.path.join(directory, '*.csv'))):
            m = re.search(season_pattern, os.path.basename(f))
            if m:
                self.sources.append((f, 'csv', group, m.group(1)))
    
    def add_zip_json_source(self, zip_path: str,
                             passing_group: str = 'PASS',
                             possessions_group: str = 'POSS'):
        """
        Extract passing + possessions data from a zip of per-game JSONs.
        
        Expected zip structure:
          {season}/passing/{date}.json
          {season}/possessions/{date}.json
        
        Each JSON is NBA API resultSets format.
        Per-game rows are aggregated to season means per player.
        """
        if not os.path.exists(zip_path):
            print(f"  Zip not found: {zip_path}")
            return
        
        frames = {'passing': [], 'possessions': []}
        season_pat = re.compile(r'(\d{4})-?(\d{2})')
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for entry in zf.namelist():
                if not entry.endswith('.json') or entry.endswith('/'):
                    continue
                
                entry_lower = entry.lower()
                if '/passing/' in entry_lower:
                    category = 'passing'
                elif '/possessions/' in entry_lower or '/possession/' in entry_lower:
                    category = 'possessions'
                else:
                    continue
                
                season_match = season_pat.search(entry)
                if not season_match:
                    continue
                
                y1, y2 = int(season_match.group(1)), int(season_match.group(2))
                season = f"{y1}-{str(y2 % 100).zfill(2)}"
                
                try:
                    df = parse_nba_api_json(json.loads(zf.read(entry)))
                    if not df.empty and len(df) >= 5:
                        df['SEASON'] = season
                        frames[category].append(df)
                except Exception:
                    continue
        
        # Aggregate to season means
        for category, group in [('passing', passing_group), 
                                 ('possessions', possessions_group)]:
            if not frames[category]:
                continue
            agg = self._aggregate_frames(frames[category], group)
            if agg is not None:
                self.sources.append(('__aggregated__', 'pre_agg', group, '__multi__'))
                # Store pre-aggregated data
                if not hasattr(self, '_pre_aggregated'):
                    self._pre_aggregated = {}
                self._pre_aggregated[group] = agg
    
    def _aggregate_frames(self, frames: List[pd.DataFrame], 
                           group: str) -> Optional[pd.DataFrame]:
        """Concat per-game frames → season means per player."""
        big = pd.concat(frames, ignore_index=True)
        
        # Find player name column
        name_col = self._detect_name_col(big)
        if name_col is None:
            return None
        
        big['PLAYER_KEY'] = big[name_col].map(canonical_norm)
        big = big.dropna(subset=['PLAYER_KEY'])
        
        # Select numeric feature columns
        prefix = f"PRIOR_{group}_"
        skip_cols = {'SEASON', 'PLAYER_KEY', name_col, 'PLAYER_ID', 
                     'TEAM_ID', 'TEAM_ABBREVIATION', 'GP', 'W', 'L'}
        
        feat_map = {}
        for c in big.columns:
            if c in skip_cols or self.SKIP_PATTERN.search(c):
                continue
            if not pd.api.types.is_numeric_dtype(big[c]):
                big[c] = pd.to_numeric(big[c], errors='coerce')
                if big[c].notna().mean() < 0.3:
                    continue
            feat_map[c] = prefix + self._clean_col(c)
        
        if not feat_map:
            return None
        
        big = big.rename(columns=feat_map)
        agg_cols = list(feat_map.values())
        result = big.groupby(['SEASON', 'PLAYER_KEY'], as_index=False)[agg_cols].mean()
        
        for c in agg_cols:
            result[c] = result[c].astype('float32')
        
        return result
    
    def build(self) -> pd.DataFrame:
        """
        Process all registered sources into a unified priors library.
        
        Returns DataFrame indexed by (SEASON, PLAYER_KEY) with all
        PRIOR_* feature columns.
        """
        priors_idx = None
        loaded, skipped = 0, 0
        
        for filepath, fmt, group, season in self.sources:
            if fmt == 'pre_agg':
                # Handle pre-aggregated data (from zip sources)
                if hasattr(self, '_pre_aggregated') and group in self._pre_aggregated:
                    agg = self._pre_aggregated[group]
                    idx = agg.set_index(['SEASON', 'PLAYER_KEY'])
                    priors_idx = idx if priors_idx is None else priors_idx.combine_first(idx)
                    loaded += 1
                continue
            
            try:
                df = (pd.read_excel(filepath, sheet_name=0) if fmt == 'xlsx' 
                      else pd.read_csv(filepath))
            except Exception:
                skipped += 1
                continue
            
            name_col = self._detect_name_col(df)
            if name_col is None:
                skipped += 1
                continue
            
            df = df.copy()
            df['SEASON'] = season
            df['PLAYER_KEY'] = df[name_col].map(canonical_norm)
            df = df.dropna(subset=['PLAYER_KEY'])
            
            if len(df) < 10:
                skipped += 1
                continue
            
            # Extract numeric features
            prefix = f"PRIOR_{group}_"
            keys = ['SEASON', 'PLAYER_KEY']
            out = df[keys].copy()
            feat_count = 0
            
            for c in df.columns:
                if self.SKIP_PATTERN.search(c) or c in keys or c == name_col:
                    continue
                vals = df[c]
                if not pd.api.types.is_numeric_dtype(vals):
                    vals = pd.to_numeric(
                        vals.astype(str).str.replace('%', '').str.replace(',', ''),
                        errors='coerce'
                    )
                    if vals.notna().mean() < 0.5:
                        continue
                out[prefix + self._clean_col(c)] = vals.astype('float32')
                feat_count += 1
            
            if feat_count == 0:
                skipped += 1
                continue
            
            # Deduplicate per player-season
            num_cols = [c for c in out.columns if c not in keys]
            out = out.groupby(keys, as_index=False)[num_cols].mean()
            
            # Drop constant columns
            nunique = out[num_cols].nunique(dropna=True)
            keep = keys + nunique[nunique > 1].index.tolist()
            out = out[keep]
            
            out_idx = out.set_index(keys)
            priors_idx = out_idx if priors_idx is None else priors_idx.combine_first(out_idx)
            loaded += 1
        
        print(f"  Loaded: {loaded}, Skipped: {skipped}")
        
        if priors_idx is None:
            raise RuntimeError("No priors built from any source!")
        
        self.priors = priors_idx.reset_index()
        self.priors = self.priors.loc[:, ~self.priors.columns.duplicated()]
        return self.priors
    
    def merge_to_main(self, main: pd.DataFrame, 
                       season_col: str = 'SEASON') -> pd.DataFrame:
        """
        Merge priors into main game-level DataFrame using PRIOR SEASON key.
        
        CRITICAL: merge on PREV_SEASON, not current SEASON.
        This prevents within-season data leakage.
        
        A player's 2023-24 stats → available for their 2024-25 games.
        2019-20 games will have NaN priors (no 2018-19 data) — expected.
        """
        if self.priors is None:
            raise RuntimeError("Call build() before merge_to_main()")
        
        main = main.copy()
        
        # Ensure PLAYER_KEY exists
        name_col = self._detect_name_col(main) or 'PLAYER'
        if 'PLAYER_KEY' not in main.columns:
            main['PLAYER_KEY'] = main[name_col].map(canonical_norm)
        
        # Compute prior season
        main['PREV_SEASON'] = main[season_col].map(prior_season)
        
        # Drop any existing PRIOR columns
        old_prior = [c for c in main.columns if c.startswith('PRIOR_')]
        if old_prior:
            main = main.drop(columns=old_prior)
        
        # Merge on PREV_SEASON — the core anti-leakage mechanism
        n_before = len(main)
        merged = main.merge(
            self.priors,
            left_on=['PREV_SEASON', 'PLAYER_KEY'],
            right_on=['SEASON', 'PLAYER_KEY'],
            how='left',
            suffixes=('', '_PRI')
        )
        
        # Clean up merge artifacts
        if 'SEASON_PRI' in merged.columns:
            merged.drop(columns='SEASON_PRI', inplace=True)
        
        assert len(merged) == n_before, f"Row explosion: {len(merged)} vs {n_before}"
        
        # NO fillna — LightGBM handles NaN natively
        return merged
    
    @staticmethod
    def _detect_name_col(df: pd.DataFrame) -> Optional[str]:
        """Find the player name column in a DataFrame."""
        for c in ['PLAYER_NAME', 'PLAYER', 'Player', 'NAME']:
            if c in df.columns:
                return c
        return None
    
    @staticmethod
    def _clean_col(raw: str) -> str:
        """Normalize column name to uppercase alphanumeric + underscores."""
        s = str(raw).upper()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^A-Z0-9_]", "", s)
        return s


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == '__main__':
    """
    builder = PriorSeasonBuilder()
    
    # Add tracking data
    builder.add_excel_sources('/content/', {
        'CatchShoot': 'CATCH', 'Drives': 'DRIVE',
        'Defense': 'DEF_TRK', 'Efficiency': 'EFF',
    })
    
    # Add season aggregates
    builder.add_csv_directory('/data/player_stats/', 'DASH_USAGE')
    builder.add_csv_directory('/data/hustle/', 'HUSTLE')
    
    # Add passing + possessions from zip
    builder.add_zip_json_source('/data/ptstats.zip')
    
    # Build priors library
    priors = builder.build()
    print(f"Built {priors.shape[1]-2} features for {priors['PLAYER_KEY'].nunique()} players")
    
    # Merge into main dataset
    main = pd.read_csv('game_data.csv')
    enriched = builder.merge_to_main(main)
    """
    print("Prior-season feature builder loaded.")
    print("302 features across 14 source groups (7 seasons).")
