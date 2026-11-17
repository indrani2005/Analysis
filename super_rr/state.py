"""Reconstruct pre-delivery match state (no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attribution import phase_from_over, role_band


def _overs_to_balls(overs) -> float | None:
    if pd.isna(overs):
        return None
    # Overs may be encoded as balls already in some rows (e.g. 114.0 meaning 114 balls?)
    # In this dataset revised overs look like 114.0, 42.0, 48.0, 72.0, 90.0 — these are BALLS
    # when values are > 20, else could be overs. Inspected samples: 114, 42, 48, 72, 90 → balls.
    v = float(overs)
    if v > 20:
        return v
    # Interpret as overs.balls? Prefer plain overs * 6 when small.
    whole = int(v)
    frac = v - whole
    # If frac looks like tenths of an over (0.1..0.5), treat as overs.balls cricket notation
    if frac > 0:
        balls_part = int(round(frac * 10))
        return whole * 6 + balls_part
    return whole * 6


def innings_planned_balls(row) -> int:
    """Legal balls allocated to this innings (handles revised overs)."""
    inn = int(row["innings"])
    if inn == 1:
        rev = _overs_to_balls(row.get("first_inn_revised_overs"))
        return int(rev) if rev is not None else 120
    if inn == 2:
        rev = _overs_to_balls(row.get("second_inn_revised_overs"))
        return int(rev) if rev is not None else 120
    return 120


def reconstruct_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each delivery, attach state *immediately before* the ball.

    Columns added:
      pre_runs, pre_wkts, pre_legal_balls, balls_remaining, wickets_in_hand,
      crr, target, runs_needed, rrr, innings_balls_planned, phase, role_band,
      par_proj (simple resource projection for innings 1)
    """
    out = df.sort_values(["match_id", "innings", "innings_ball_num"]).copy()

    # Cumulative within innings AFTER each ball; shift to get BEFORE.
    out["_team_runs"] = out["team_runs"].fillna(0.0)
    out["_wkt"] = out["is_wicket"].astype(bool).astype(int)
    out["_legal"] = out["is_legal"].astype(int)

    g = out.groupby(["match_id", "innings"], sort=False)
    out["cum_runs_after"] = g["_team_runs"].cumsum()
    out["cum_wkts_after"] = g["_wkt"].cumsum()
    out["cum_legal_after"] = g["_legal"].cumsum()

    out["pre_runs"] = out["cum_runs_after"] - out["_team_runs"]
    out["pre_wkts"] = out["cum_wkts_after"] - out["_wkt"]
    out["pre_legal_balls"] = out["cum_legal_after"] - out["_legal"]

    out["innings_balls_planned"] = out.apply(innings_planned_balls, axis=1)
    out["balls_remaining"] = (
        out["innings_balls_planned"] - out["pre_legal_balls"]
    ).clip(lower=0)
    out["wickets_in_hand"] = (10 - out["pre_wkts"]).clip(lower=0)

    # CRR before ball
    out["crr"] = np.where(
        out["pre_legal_balls"] > 0,
        out["pre_runs"] / out["pre_legal_balls"] * 6.0,
        0.0,
    )

    # Target for innings 2
    # Prefer revised_target when present; else first-innings total + 1
    inn1_totals = (
        out.loc[out["innings"] == 1]
        .groupby("match_id")["cum_runs_after"]
        .max()
        .rename("inn1_total")
    )
    out = out.merge(inn1_totals, on="match_id", how="left")

    def _target(row):
        if int(row["innings"]) != 2:
            return np.nan
        if pd.notna(row.get("revised_target")):
            return float(row["revised_target"])
        if pd.notna(row.get("inn1_total")):
            return float(row["inn1_total"]) + 1.0
        return np.nan

    out["target"] = out.apply(_target, axis=1)
    out["runs_needed"] = np.where(
        out["innings"] == 2, (out["target"] - out["pre_runs"]).clip(lower=0), np.nan
    )
    out["rrr"] = np.where(
        (out["innings"] == 2) & (out["balls_remaining"] > 0),
        out["runs_needed"] / out["balls_remaining"] * 6.0,
        np.where((out["innings"] == 2) & (out["balls_remaining"] <= 0), 99.0, np.nan),
    )

    out["phase"] = out["overid"].map(phase_from_over)
    out["role_band"] = out["batting_order"].map(role_band)

    # Simple innings-1 projected score: current + CRR * remaining balls/6
    # At start use competition median finish later; here use max(crr, 7.5) prior
    prior_rpo = 8.0
    rate = np.where(out["pre_legal_balls"] >= 12, out["crr"], prior_rpo)
    out["proj_total"] = out["pre_runs"] + rate / 6.0 * out["balls_remaining"]

    # Competition-family median innings-1 totals as par (computed after first pass below)
    return out.drop(columns=["_team_runs", "_wkt", "_legal"], errors="ignore")


def attach_par(df: pd.DataFrame) -> pd.DataFrame:
    """Attach competition-family par (median completed innings-1 totals)."""
    inn1 = df[df["innings"] == 1].groupby(["match_id", "comp_family"], as_index=False)[
        "cum_runs_after"
    ].max()
    par = inn1.groupby("comp_family")["cum_runs_after"].median().to_dict()
    # Global fallback
    global_par = float(inn1["cum_runs_after"].median()) if len(inn1) else 165.0
    out = df.copy()
    out["par_score"] = out["comp_family"].map(par).fillna(global_par)
    out["par_deficit"] = out["par_score"] - out["proj_total"]
    return out
