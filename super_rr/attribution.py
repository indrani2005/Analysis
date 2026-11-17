"""Legal deliveries, extras, wicket credit, and competition family helpers."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    BOWLER_EXTRAS,
    BOWLER_WICKET_TYPES,
    ILLEGAL_EXTRAS,
    IPL_NAME_PATTERN,
    NON_BOWLER_EXTRAS,
)


def _norm_extra(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def is_legal_delivery(extras_type) -> bool:
    """A delivery is legal if it is not a wide or no-ball (Overstep)."""
    return _norm_extra(extras_type) not in ILLEGAL_EXTRAS


def bat_runs(runs) -> float:
    """Off-the-bat runs credited to the striker (dataset `runs` column)."""
    if pd.isna(runs):
        return 0.0
    return float(runs)


def total_team_runs(runs, extras_type, extras1_runs, extras_2_runs=0) -> float:
    """Runs added to the batting team total on this delivery."""
    r = bat_runs(runs)
    e1 = 0.0 if pd.isna(extras1_runs) else float(extras1_runs)
    e2 = 0.0 if pd.isna(extras_2_runs) else float(extras_2_runs)
    # Dataset convention: bat runs in `runs`; extras in extras1/2.
    # For wides/byes/legbyes, runs is typically 0.
    return r + e1 + e2


def bowler_runs_conceded(runs, extras_type, extras1_runs, extras_2_runs=0) -> float:
    """
    Runs charged against the bowler.

    - Off-the-bat runs always charged.
    - Wides / Overstep (no-balls): extras charged to bowler.
    - Byes / leg-byes / penalty / other: NOT charged to bowler.
    """
    r = bat_runs(runs)
    extra = _norm_extra(extras_type)
    e1 = 0.0 if pd.isna(extras1_runs) else float(extras1_runs)
    e2 = 0.0 if pd.isna(extras_2_runs) else float(extras_2_runs)
    if extra in BOWLER_EXTRAS:
        return r + e1 + e2
    if extra in NON_BOWLER_EXTRAS:
        return r  # usually 0 for pure byes/legbyes
    # No extras tag: bat runs only
    return r


def is_bowler_wicket(is_wicket, dismissal_type) -> bool:
    """True if the wicket should be credited to the bowler."""
    if not bool(is_wicket):
        return False
    if pd.isna(dismissal_type):
        return False
    dt = str(dismissal_type).strip().lower().replace("_", " ")
    # Normalise common variants
    dt = dt.replace("obstracting", "obstructing")
    if dt in BOWLER_WICKET_TYPES:
        return True
    # Allow "caught & bowled" style if present
    if "caught" in dt and "run out" not in dt:
        return True
    return False


def is_batter_dismissal(is_wicket, dismissed_player_id, striker_id) -> bool:
    """Whether the striker was dismissed on this ball (for batter dismissal rate)."""
    if not bool(is_wicket):
        return False
    if pd.isna(dismissed_player_id) or pd.isna(striker_id):
        # If unknown dismissed id but wicket flagged, attribute to striker conservatively
        # only when dismissal is not clearly a non-striker event — handled by caller if needed.
        return True
    return int(dismissed_player_id) == int(striker_id)


def competition_family(comp_name: str) -> str:
    """Map competition name to IPL vs T20I family."""
    if pd.isna(comp_name):
        return "T20I"
    if re.search(IPL_NAME_PATTERN, str(comp_name), flags=re.IGNORECASE):
        return "IPL"
    return "T20I"


def role_band(batting_order) -> str:
    if pd.isna(batting_order):
        return "middle"
    o = int(batting_order)
    if o <= 3:
        return "top"
    if o <= 6:
        return "middle"
    return "lower"


def phase_from_over(overid) -> str:
    if pd.isna(overid):
        return "middle"
    o = int(overid)
    if o <= 6:
        return "powerplay"
    if o <= 15:
        return "middle"
    return "death"


def annotate_deliveries(bbb: pd.DataFrame) -> pd.DataFrame:
    """Add attribution columns to ball-by-ball frame."""
    df = bbb.copy()
    extras = df["extras_type"].map(_norm_extra)
    df["extras_norm"] = extras
    df["is_legal"] = ~extras.isin(ILLEGAL_EXTRAS)
    df["bat_runs"] = df["runs"].fillna(0).astype(float)
    df["team_runs"] = [
        total_team_runs(r, e, e1, e2)
        for r, e, e1, e2 in zip(
            df["runs"], df["extras_type"], df["extras1_runs"], df["extras_2_runs"]
        )
    ]
    df["bowl_runs"] = [
        bowler_runs_conceded(r, e, e1, e2)
        for r, e, e1, e2 in zip(
            df["runs"], df["extras_type"], df["extras1_runs"], df["extras_2_runs"]
        )
    ]
    df["is_bowler_wicket"] = [
        is_bowler_wicket(w, d) for w, d in zip(df["is_wicket"], df["dismissal_type"])
    ]
    df["is_striker_out"] = [
        is_batter_dismissal(w, did, sid)
        for w, did, sid in zip(
            df["is_wicket"], df["dismissed_player_id"], df["striker_id"]
        )
    ]
    # Non-striker run-outs: striker not out
    runout_mask = df["dismissal_type"].fillna("").astype(str).str.lower().str.contains(
        "runout|run out"
    )
    non_striker_out = runout_mask & (
        df["dismissed_player_id"].fillna(-1).astype(int) != df["striker_id"].fillna(-2).astype(int)
    )
    df.loc[non_striker_out, "is_striker_out"] = False

    df["is_boundary"] = (df["is_four"].astype(bool) | df["is_six"].astype(bool)) & (
        df["bat_runs"] > 0
    )
    df["is_dot_bat"] = (df["bat_runs"] == 0) & df["is_legal"] & (~df["is_striker_out"])
    return df


def resolve_player_names(map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical player_id → display name.

    Uses the most frequent name per ID in the mapping table.
    Never merges distinct IDs that share a display name.
    """
    g = (
        map_df.groupby(["player_id", "player_name"], as_index=False)
        .size()
        .sort_values(["player_id", "size"], ascending=[True, False])
    )
    canon = g.drop_duplicates("player_id", keep="first")[
        ["player_id", "player_name"]
    ].rename(columns={"player_name": "player_name_canon"})
    return canon


def find_name_collisions(map_df: pd.DataFrame) -> pd.DataFrame:
    """Names that map to more than one player_id (must not be merged)."""
    s = map_df.groupby("player_name")["player_id"].nunique()
    bad = s[s > 1].index
    return (
        map_df[map_df["player_name"].isin(bad)]
        .groupby("player_name")["player_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="player_ids")
    )
