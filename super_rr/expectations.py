"""Role- and context-adjusted expected performance (Empirical Bayes cells)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EB_PRIOR_BALLS


def _rrr_band(rrr) -> str:
    if pd.isna(rrr):
        return "na"
    x = float(rrr)
    if x < 7:
        return "lt7"
    if x < 9:
        return "7_9"
    if x < 11:
        return "9_11"
    if x < 14:
        return "11_14"
    return "ge14"


def _wkt_band(pre_wkts) -> str:
    w = int(pre_wkts) if pd.notna(pre_wkts) else 0
    if w <= 1:
        return "0_1"
    if w <= 3:
        return "2_3"
    if w <= 5:
        return "4_5"
    return "6p"


def _par_band(par_deficit) -> str:
    if pd.isna(par_deficit):
        return "na"
    x = float(par_deficit)
    if x < -15:
        return "ahead"
    if x < 15:
        return "par"
    return "behind"


def attach_context_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rrr_band"] = out["rrr"].map(_rrr_band)
    out["wkt_band"] = out["pre_wkts"].map(_wkt_band)
    out["par_band"] = out["par_deficit"].map(_par_band)
    # Batter cell: family × innings × phase × wkt × role × (rrr or par band)
    out["bat_cell"] = np.where(
        out["innings"] == 2,
        out["comp_family"].astype(str)
        + "|2|"
        + out["phase"].astype(str)
        + "|"
        + out["wkt_band"]
        + "|"
        + out["role_band"].astype(str)
        + "|"
        + out["rrr_band"],
        out["comp_family"].astype(str)
        + "|1|"
        + out["phase"].astype(str)
        + "|"
        + out["wkt_band"]
        + "|"
        + out["role_band"].astype(str)
        + "|"
        + out["par_band"],
    )
    # Bowler cell: family × innings × phase × wkt × pace/spin × (rrr/par)
    pos = out.get("pace_or_spin", pd.Series("UNK", index=out.index)).fillna("UNK")
    out["bowl_cell"] = np.where(
        out["innings"] == 2,
        out["comp_family"].astype(str)
        + "|2|"
        + out["phase"].astype(str)
        + "|"
        + out["wkt_band"]
        + "|"
        + pos.astype(str)
        + "|"
        + out["rrr_band"],
        out["comp_family"].astype(str)
        + "|1|"
        + out["phase"].astype(str)
        + "|"
        + out["wkt_band"]
        + "|"
        + pos.astype(str)
        + "|"
        + out["par_band"],
    )
    return out


def _eb_mean(sum_x: pd.Series, n: pd.Series, prior_mean: float, k: float) -> pd.Series:
    return (sum_x + k * prior_mean) / (n + k)


def compute_expected_outcomes(df: pd.DataFrame, k: float = EB_PRIOR_BALLS) -> pd.DataFrame:
    """
    Add expected bat runs / bowl runs / bowl wicket rate by context cell,
    shrunk toward competition-family priors.
    """
    out = attach_context_keys(df)

    # Global / family priors on legal balls primarily; still score illegal for bowl extras
    legal = out["is_legal"]

    # --- Batting expectations on legal balls (bat_runs per ball) ---
    fam_bat_prior = (
        out.loc[legal]
        .groupby("comp_family")["bat_runs"]
        .mean()
        .to_dict()
    )
    cell_bat = (
        out.loc[legal]
        .groupby("bat_cell")
        .agg(sum_runs=("bat_runs", "sum"), n=("bat_runs", "size"), fam=("comp_family", "first"))
    )
    cell_bat["prior"] = cell_bat["fam"].map(fam_bat_prior).fillna(out.loc[legal, "bat_runs"].mean())
    cell_bat["exp_bat"] = _eb_mean(cell_bat["sum_runs"], cell_bat["n"], cell_bat["prior"], k)
    out["exp_bat_runs"] = out["bat_cell"].map(cell_bat["exp_bat"])
    # fallback
    out["exp_bat_runs"] = out["exp_bat_runs"].fillna(
        out["comp_family"].map(fam_bat_prior)
    ).fillna(out["bat_runs"].mean())

    # Dismissal expectation (striker outs on legal balls)
    cell_out = (
        out.loc[legal]
        .groupby("bat_cell")
        .agg(sum_out=("is_striker_out", "sum"), n=("is_striker_out", "size"), fam=("comp_family", "first"))
    )
    fam_out_prior = out.loc[legal].groupby("comp_family")["is_striker_out"].mean().to_dict()
    cell_out["prior"] = cell_out["fam"].map(fam_out_prior).fillna(out.loc[legal, "is_striker_out"].mean())
    cell_out["exp_out"] = _eb_mean(cell_out["sum_out"], cell_out["n"], cell_out["prior"], k)
    out["exp_dismiss"] = out["bat_cell"].map(cell_out["exp_out"]).fillna(
        out["comp_family"].map(fam_out_prior)
    ).fillna(0.03)

    # --- Bowling expectations (bowl_runs on all deliveries bowler bowls; wickets on legal) ---
    fam_bowl_prior = out.groupby("comp_family")["bowl_runs"].mean().to_dict()
    cell_bowl = (
        out.groupby("bowl_cell")
        .agg(sum_runs=("bowl_runs", "sum"), n=("bowl_runs", "size"), fam=("comp_family", "first"))
    )
    cell_bowl["prior"] = cell_bowl["fam"].map(fam_bowl_prior).fillna(out["bowl_runs"].mean())
    cell_bowl["exp_bowl"] = _eb_mean(cell_bowl["sum_runs"], cell_bowl["n"], cell_bowl["prior"], k)
    out["exp_bowl_runs"] = out["bowl_cell"].map(cell_bowl["exp_bowl"]).fillna(
        out["comp_family"].map(fam_bowl_prior)
    ).fillna(out["bowl_runs"].mean())

    cell_bw = (
        out.loc[legal]
        .groupby("bowl_cell")
        .agg(sum_w=("is_bowler_wicket", "sum"), n=("is_bowler_wicket", "size"), fam=("comp_family", "first"))
    )
    fam_bw_prior = out.loc[legal].groupby("comp_family")["is_bowler_wicket"].mean().to_dict()
    cell_bw["prior"] = cell_bw["fam"].map(fam_bw_prior).fillna(out.loc[legal, "is_bowler_wicket"].mean())
    cell_bw["exp_w"] = _eb_mean(cell_bw["sum_w"], cell_bw["n"], cell_bw["prior"], k)
    out["exp_bowl_wkt"] = out["bowl_cell"].map(cell_bw["exp_w"]).fillna(
        out["comp_family"].map(fam_bw_prior)
    ).fillna(0.04)

    # Residuals (positive = good for that role)
    # Batter: runs above expected - wicket penalty
    WICKET_RUN_VALUE = 14.0  # approximate T20 wicket cost in runs
    out["bat_residual"] = (out["bat_runs"] - out["exp_bat_runs"]) - WICKET_RUN_VALUE * (
        out["is_striker_out"].astype(float) - out["exp_dismiss"]
    )
    # Only meaningful on balls faced (legal for SR-like); still compute for all and mask later
    out["bowl_residual"] = (out["exp_bowl_runs"] - out["bowl_runs"]) + WICKET_RUN_VALUE * (
        out["is_bowler_wicket"].astype(float) - out["exp_bowl_wkt"]
    )
    return out
