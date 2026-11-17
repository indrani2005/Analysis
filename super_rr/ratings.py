"""Player ratings: competition-normalized composites + bootstrap stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    BOOTSTRAP_SEED,
    COMP_WEIGHT_SCHEMES,
    COMPOSITE_WEIGHTS,
    EB_PRIOR_BALLS,
    ELIGIBILITY_PRIMARY,
    N_BOOTSTRAP,
    PRIMARY_COMP_SCHEME,
)


@dataclass
class RankConfig:
    min_balls: int = ELIGIBILITY_PRIMARY[0]
    min_matches: int = ELIGIBILITY_PRIMARY[1]
    comp_scheme: str = PRIMARY_COMP_SCHEME
    n_bootstrap: int = N_BOOTSTRAP
    seed: int = BOOTSTRAP_SEED


def _safe_z(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd is None or sd < 1e-9:
        return s * 0.0
    return (s - mu) / sd


def _eb_shrink(mean: float, n: float, prior: float, k: float = EB_PRIOR_BALLS) -> float:
    return (n * mean + k * prior) / (n + k)


def player_match_aggregates(
    df: pd.DataFrame, role: str
) -> pd.DataFrame:
    """
    role: 'batter' or 'bowler'
    Returns match × player × competition family aggregates on pressure / non-pressure.
    """
    if role == "batter":
        # Batters: legal balls faced
        sub = df[df["is_legal"]].copy()
        pid = "striker_id"
        pname = "striker_name_canon"
        resid = "bat_residual"
    else:
        sub = df.copy()
        pid = "bowler_id"
        pname = "bowler_name_canon"
        resid = "bowl_residual"

    sub = sub.rename(columns={pid: "player_id", pname: "player_name"})
    sub["w"] = np.where(
        sub["is_pressure"], np.maximum(sub["pressure_intensity"], 0.25), 0.0
    )
    sub["w_resid"] = sub[resid] * sub["w"]
    gcols = ["player_id", "player_name", "match_id", "comp_family"]
    run_col = "bat_runs" if role == "batter" else "bowl_runs"
    exp_col = "exp_bat_runs" if role == "batter" else "exp_bowl_runs"

    base = sub.groupby(gcols, as_index=False).agg(
        balls=("player_id", "size"),
        pressure_balls=("is_pressure", "sum"),
    )
    pr = (
        sub[sub["is_pressure"]]
        .groupby(gcols, as_index=False)
        .agg(
            p_resid_mean=(resid, "mean"),
            p_resid_wsum=("w_resid", "sum"),
            p_wsum=("w", "sum"),
            p_balls=("is_pressure", "size"),
            p_runs=(run_col, "sum"),
            p_exp=(exp_col, "sum"),
        )
    )
    npr = (
        sub[~sub["is_pressure"]]
        .groupby(gcols, as_index=False)
        .agg(np_resid_mean=(resid, "mean"), np_balls=(resid, "size"))
    )
    out = base.merge(pr, on=gcols, how="left").merge(npr, on=gcols, how="left")
    out["p_resid_wmean"] = np.where(
        out["p_wsum"] > 0,
        out["p_resid_wsum"] / out["p_wsum"],
        out["p_resid_mean"],
    )
    out["clutch_delta"] = out["p_resid_wmean"] - out["np_resid_mean"]
    out["positive_pressure_match"] = (out["p_resid_wmean"] > 0).astype(float)
    return out


def _combine_family_scores(
    fam_df: pd.DataFrame, scheme: str
) -> pd.DataFrame:
    """
    fam_df indexed with player-level metrics per comp_family.
    Columns required: player_id, player_name, comp_family, rae, clutch, consistency,
    pressure_balls, pressure_matches
    """
    rows = []
    for pid, g in fam_df.groupby("player_id"):
        name = g["player_name"].iloc[0]
        parts = {}
        for _, r in g.iterrows():
            parts[r["comp_family"]] = r
        families = [f for f in ("IPL", "T20I") if f in parts]
        if not families:
            continue
        if scheme == "equal":
            weights = {f: 1.0 / len(families) for f in families}
        else:  # sample_proportional
            tot = sum(float(parts[f]["pressure_balls"]) for f in families)
            weights = {
                f: (float(parts[f]["pressure_balls"]) / tot if tot > 0 else 1.0 / len(families))
                for f in families
            }
        rae = sum(weights[f] * float(parts[f]["rae_z"]) for f in families)
        clutch = sum(weights[f] * float(parts[f]["clutch_z"]) for f in families)
        cons = sum(weights[f] * float(parts[f]["consistency"]) for f in families)
        n_balls = sum(float(parts[f]["pressure_balls"]) for f in families)
        n_matches = sum(float(parts[f]["pressure_matches"]) for f in families)
        rows.append(
            {
                "player_id": pid,
                "player_name": name,
                "rae": rae,
                "clutch": clutch,
                "consistency": cons,
                "pressure_balls": n_balls,
                "pressure_matches": n_matches,
                "n_families": len(families),
                "families": ",".join(families),
                "w_ipl": weights.get("IPL", 0.0),
                "w_t20i": weights.get("T20I", 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_family_player_metrics(match_agg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match-level rows to player × family metrics with within-family z."""
    # Pressure appearances only
    m = match_agg[match_agg["p_balls"].fillna(0) > 0].copy()
    rows = []
    for (pid, fam), g in m.groupby(["player_id", "comp_family"]):
        n_balls = float(g["p_balls"].sum())
        n_matches = int(g["match_id"].nunique())
        # EB shrink mean residual
        raw_rae = float(np.average(g["p_resid_wmean"].fillna(0), weights=g["p_balls"]))
        rae = _eb_shrink(raw_rae, n_balls, 0.0, EB_PRIOR_BALLS)
        raw_clutch = float(
            np.average(g["clutch_delta"].fillna(0), weights=g["p_balls"].clip(lower=1))
        )
        clutch = _eb_shrink(raw_clutch, n_balls, 0.0, EB_PRIOR_BALLS)
        consistency = float(g["positive_pressure_match"].mean())
        rows.append(
            {
                "player_id": pid,
                "player_name": g["player_name"].iloc[0],
                "comp_family": fam,
                "rae": rae,
                "clutch": clutch,
                "consistency": consistency,
                "pressure_balls": n_balls,
                "pressure_matches": n_matches,
            }
        )
    fam = pd.DataFrame(rows)
    if fam.empty:
        return fam
    # Within-family z for rae/clutch
    fam["rae_z"] = fam.groupby("comp_family")["rae"].transform(_safe_z)
    fam["clutch_z"] = fam.groupby("comp_family")["clutch"].transform(_safe_z)
    return fam


def finalize_scores(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined
    out = combined.copy()
    out["cons_z"] = _safe_z(out["consistency"])
    out["R_mean"] = (
        COMPOSITE_WEIGHTS["rae"] * out["rae"]
        + COMPOSITE_WEIGHTS["clutch"] * out["clutch"]
        + COMPOSITE_WEIGHTS["consistency"] * out["cons_z"]
    )
    # Additional EB reliability toward 0 by sample size
    out["R_shrunk"] = [
        _eb_shrink(r, n, 0.0, EB_PRIOR_BALLS)
        for r, n in zip(out["R_mean"], out["pressure_balls"])
    ]
    return out


def eligible_mask(df: pd.DataFrame, min_balls: int, min_matches: int) -> pd.Series:
    return (df["pressure_balls"] >= min_balls) & (df["pressure_matches"] >= min_matches)


def bootstrap_stable_ranks(
    match_agg: pd.DataFrame,
    cfg: RankConfig,
) -> pd.DataFrame:
    """
    Match-cluster bootstrap of combined scores.

    Primary metric: R_stable = mean(R_shrunk) - sd(R_shrunk)
    """
    rng = np.random.default_rng(cfg.seed)
    matches = match_agg["match_id"].unique()
    # Point estimate on full data
    fam = build_family_player_metrics(match_agg)
    comb = _combine_family_scores(fam, cfg.comp_scheme)
    point = finalize_scores(comb)
    point = point.loc[eligible_mask(point, cfg.min_balls, cfg.min_matches)].copy()
    if point.empty:
        return point

    # Bootstrap distributions
    boot_scores = {int(pid): [] for pid in point["player_id"]}
    n_boot = cfg.n_bootstrap
    for b in range(n_boot):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        # Weight each match by multiplicity
        vc = pd.Series(sampled).value_counts()
        m = match_agg[match_agg["match_id"].isin(vc.index)].copy()
        m["_rep"] = m["match_id"].map(vc).astype(int)
        # Expand weights into weighted aggregation via repeat index (efficient-ish)
        # For speed: multiply p_balls by rep when averaging
        m = m.copy()
        m["p_balls_w"] = m["p_balls"].fillna(0) * m["_rep"]
        m["p_resid_wmean_w"] = m["p_resid_wmean"].fillna(0) * m["p_balls_w"]
        m["clutch_w"] = m["clutch_delta"].fillna(0) * m["p_balls_w"]
        m["pos_w"] = m["positive_pressure_match"].fillna(0) * m["_rep"]

        rows = []
        for (pid, fam_), g in m.groupby(["player_id", "comp_family"]):
            n_balls = float(g["p_balls_w"].sum())
            if n_balls <= 0:
                continue
            n_matches = float(g["_rep"].sum())  # weighted match count
            rae = _eb_shrink(
                float(g["p_resid_wmean_w"].sum() / n_balls), n_balls, 0.0, EB_PRIOR_BALLS
            )
            clutch = _eb_shrink(
                float(g["clutch_w"].sum() / n_balls), n_balls, 0.0, EB_PRIOR_BALLS
            )
            consistency = float(g["pos_w"].sum() / max(n_matches, 1e-9))
            rows.append(
                {
                    "player_id": pid,
                    "player_name": g["player_name"].iloc[0],
                    "comp_family": fam_,
                    "rae": rae,
                    "clutch": clutch,
                    "consistency": consistency,
                    "pressure_balls": n_balls,
                    "pressure_matches": n_matches,
                }
            )
        fam_b = pd.DataFrame(rows)
        if fam_b.empty:
            continue
        fam_b["rae_z"] = fam_b.groupby("comp_family")["rae"].transform(_safe_z)
        fam_b["clutch_z"] = fam_b.groupby("comp_family")["clutch"].transform(_safe_z)
        comb_b = _combine_family_scores(fam_b, cfg.comp_scheme)
        sc = finalize_scores(comb_b)
        sc = sc.loc[eligible_mask(sc, cfg.min_balls, cfg.min_matches)]
        for pid, val in zip(sc["player_id"], sc["R_shrunk"]):
            pid = int(pid)
            if pid in boot_scores:
                boot_scores[pid].append(float(val))

    stats = []
    for _, row in point.iterrows():
        pid = int(row["player_id"])
        arr = np.array(boot_scores.get(pid, []), dtype=float)
        if len(arr) == 0:
            mu, sd = float(row["R_shrunk"]), 0.0
            p10 = mu
        else:
            mu = float(arr.mean())
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            p10 = float(np.quantile(arr, 0.10))
        r_stable = mu - sd
        stats.append(
            {
                **row.to_dict(),
                "boot_mean": mu,
                "boot_sd": sd,
                "R_stable": r_stable,
                "R_lcb90": p10,
                "boot_n": int(len(arr)),
            }
        )
    out = pd.DataFrame(stats)
    out = out.sort_values("R_stable", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    # Top-10 frequency
    # Recompute quickly from stored boot arrays via rank in each boot — approximate using score thresholds
    # Full rank frequency: redo light pass
    out["top10_freq"] = 0.0
    if n_boot > 0:
        # Build per-bootstrap ranking among eligible point players
        # Using collected scores: for each bootstrap index alignment is ragged — recompute freq via score quantiles
        # Simpler approach: for each player, fraction of boots where score >= 10th best mean score in that boot
        # We didn't store aligned boots; estimate via comparing each player's boot draws to overall
        # Re-run a compact frequency using point players only:
        pass
    return out


def top10_frequency(
    match_agg: pd.DataFrame, cfg: RankConfig, player_ids: list[int]
) -> dict[int, float]:
    """Fraction of bootstrap resamples in which each player finishes top-10 by R_shrunk."""
    rng = np.random.default_rng(cfg.seed + 7)
    matches = match_agg["match_id"].unique()
    hits = {int(p): 0 for p in player_ids}
    total = 0
    for _ in range(cfg.n_bootstrap):
        sampled = rng.choice(matches, size=len(matches), replace=True)
        vc = pd.Series(sampled).value_counts()
        m = match_agg[match_agg["match_id"].isin(vc.index)].copy()
        m["_rep"] = m["match_id"].map(vc).astype(int)
        m["p_balls_w"] = m["p_balls"].fillna(0) * m["_rep"]
        m["p_resid_wmean_w"] = m["p_resid_wmean"].fillna(0) * m["p_balls_w"]
        m["clutch_w"] = m["clutch_delta"].fillna(0) * m["p_balls_w"]
        m["pos_w"] = m["positive_pressure_match"].fillna(0) * m["_rep"]
        rows = []
        for (pid, fam_), g in m.groupby(["player_id", "comp_family"]):
            n_balls = float(g["p_balls_w"].sum())
            if n_balls <= 0:
                continue
            n_matches = float(g["_rep"].sum())
            rae = _eb_shrink(float(g["p_resid_wmean_w"].sum() / n_balls), n_balls, 0.0, EB_PRIOR_BALLS)
            clutch = _eb_shrink(float(g["clutch_w"].sum() / n_balls), n_balls, 0.0, EB_PRIOR_BALLS)
            consistency = float(g["pos_w"].sum() / max(n_matches, 1e-9))
            rows.append(
                {
                    "player_id": pid,
                    "player_name": g["player_name"].iloc[0],
                    "comp_family": fam_,
                    "rae": rae,
                    "clutch": clutch,
                    "consistency": consistency,
                    "pressure_balls": n_balls,
                    "pressure_matches": n_matches,
                }
            )
        fam_b = pd.DataFrame(rows)
        if fam_b.empty:
            continue
        fam_b["rae_z"] = fam_b.groupby("comp_family")["rae"].transform(_safe_z)
        fam_b["clutch_z"] = fam_b.groupby("comp_family")["clutch"].transform(_safe_z)
        comb_b = finalize_scores(_combine_family_scores(fam_b, cfg.comp_scheme))
        comb_b = comb_b.loc[eligible_mask(comb_b, cfg.min_balls, cfg.min_matches)]
        top = set(comb_b.sort_values("R_shrunk", ascending=False).head(10)["player_id"].astype(int))
        total += 1
        for p in hits:
            if p in top:
                hits[p] += 1
    if total == 0:
        return {p: 0.0 for p in hits}
    return {p: hits[p] / total for p in hits}
