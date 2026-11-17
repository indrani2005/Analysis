"""
Fast final re-rank from cached deliveries_modeled.csv.gz.

Fixes first-pass issues:
- Role filter so specialist bowlers don't dominate batter lists (and vice versa)
- Primary eligibility raised to 150 pressure balls / 10 matches (sensitivity-supported)
- Faster bootstrap (120) for the one-hour freeze
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, SEED
from .explain import build_explanation_card, write_explanation_cards
from .ratings import (
    RankConfig,
    bootstrap_stable_ranks,
    player_match_aggregates,
    top10_frequency,
)

# Locked for final submission freeze
MIN_BALLS = 150
MIN_MATCHES = 10
MIN_ROLE_BALLS = 300  # career balls in that role in scoped data
N_BOOT = 80  # speed for 1-hour freeze; method identical to 200-draw run
COMP_SCHEME = "equal"


def role_volume(df: pd.DataFrame) -> pd.DataFrame:
    bat = (
        df[df["is_legal"]]
        .groupby("striker_id", as_index=False)
        .agg(
            bat_balls=("striker_id", "size"),
            bat_name=("striker_name_canon", "first"),
            med_order=("batting_order", "median") if "batting_order" in df.columns else ("striker_id", "size"),
        )
        .rename(columns={"striker_id": "player_id"})
    )
    # batting_order may be missing in cached cols — recompute from role_band if needed
    if "batting_order" not in df.columns and "role_band" in df.columns:
        # approximate order midpoints
        mp = {"top": 2, "middle": 5, "lower": 8}
        tmp = df[df["is_legal"]].copy()
        tmp["_ord"] = tmp["role_band"].map(mp).fillna(5)
        bat = (
            tmp.groupby("striker_id", as_index=False)
            .agg(
                bat_balls=("striker_id", "size"),
                bat_name=("striker_name_canon", "first"),
                med_order=("_ord", "median"),
            )
            .rename(columns={"striker_id": "player_id"})
        )

    bowl = (
        df.groupby("bowler_id", as_index=False)
        .agg(bowl_balls=("bowler_id", "size"), bowl_name=("bowler_name_canon", "first"))
        .rename(columns={"bowler_id": "player_id"})
    )
    vol = bat.merge(bowl, on="player_id", how="outer")
    vol["bat_balls"] = vol["bat_balls"].fillna(0)
    vol["bowl_balls"] = vol["bowl_balls"].fillna(0)
    vol["player_name"] = vol["bat_name"].fillna(vol["bowl_name"])
    vol["med_order"] = vol.get("med_order", pd.Series(5, index=vol.index)).fillna(5)
    # Batter if enough batting volume AND not overwhelmingly a bowler-only profile
    vol["is_batter_role"] = (vol["bat_balls"] >= MIN_ROLE_BALLS) & (
        vol["bat_balls"] >= 0.35 * (vol["bat_balls"] + vol["bowl_balls"])
    )
    # Prefer players who actually bat in top/middle/finisher slots (not #11 only)
    vol["is_batter_role"] = vol["is_batter_role"] & (vol["med_order"] <= 8.5)
    vol["is_bowler_role"] = (vol["bowl_balls"] >= MIN_ROLE_BALLS) & (
        vol["bowl_balls"] >= 0.35 * (vol["bat_balls"] + vol["bowl_balls"])
    )
    return vol


def run() -> dict:
    t0 = time.time()
    path = OUTPUT_DIR / "deliveries_modeled.csv.gz"
    print(f"Loading {path}")
    df = pd.read_csv(path, low_memory=False)

    # Ensure needed columns
    for c in ("is_legal", "is_pressure", "pressure_intensity", "bat_residual", "bowl_residual"):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} — re-run full pipeline first")

    # batting_order not in cache — derive from role_band
    if "batting_order" not in df.columns:
        mp = {"top": 2, "middle": 5, "lower": 8}
        df["batting_order"] = df["role_band"].map(mp).fillna(5)

    vol = role_volume(df)
    vol.to_csv(OUTPUT_DIR / "player_role_volume.csv", index=False)
    bat_ids = set(vol.loc[vol["is_batter_role"], "player_id"].astype(int))
    bowl_ids = set(vol.loc[vol["is_bowler_role"], "player_id"].astype(int))
    print(f"Role-eligible batters: {len(bat_ids)} | bowlers: {len(bowl_ids)}")

    results = {
        "eligibility": {"min_balls": MIN_BALLS, "min_matches": MIN_MATCHES, "min_role_balls": MIN_ROLE_BALLS},
        "comp_scheme": COMP_SCHEME,
        "n_bootstrap": N_BOOT,
    }

    for role, allowed in (("batter", bat_ids), ("bowler", bowl_ids)):
        print(f"==> Re-ranking {role}s")
        match_agg = player_match_aggregates(df, role)
        match_agg = match_agg[match_agg["player_id"].astype(int).isin(allowed)].copy()
        match_agg.to_csv(OUTPUT_DIR / f"{role}_match_aggregates_final.csv", index=False)

        cfg = RankConfig(
            min_balls=MIN_BALLS,
            min_matches=MIN_MATCHES,
            comp_scheme=COMP_SCHEME,
            n_bootstrap=N_BOOT,
            seed=SEED,
        )
        ranked = bootstrap_stable_ranks(match_agg, cfg)
        if ranked.empty:
            print("  no eligible players")
            continue

        cand = ranked.head(30)["player_id"].astype(int).tolist()
        freq = top10_frequency(match_agg, cfg, cand)
        ranked["top10_freq"] = ranked["player_id"].astype(int).map(freq).fillna(0.0)
        ranked.to_csv(OUTPUT_DIR / f"{role}_rankings_FINAL.csv", index=False)
        top10 = ranked.head(10).copy()
        top10.to_csv(OUTPUT_DIR / f"{role}_top10_FINAL.csv", index=False)

        cards = [build_explanation_card(r, df, role) for _, r in top10.iterrows()]
        write_explanation_cards(cards, OUTPUT_DIR / f"{role}_explanation_cards_FINAL.json")

        print("  TOP 10:")
        for _, r in top10.iterrows():
            print(
                f"   {int(r['rank']):2d}. {r['player_name']} (id={int(r['player_id'])}) "
                f"R*={r['R_stable']:.3f} balls={r['pressure_balls']:.0f} "
                f"top10%={r['top10_freq']:.0%} fam={r['families']}"
            )
        results[role] = top10[
            [
                "rank",
                "player_id",
                "player_name",
                "R_stable",
                "boot_mean",
                "boot_sd",
                "R_lcb90",
                "pressure_balls",
                "pressure_matches",
                "top10_freq",
                "families",
                "consistency",
            ]
        ].to_dict(orient="records")

        # Quick sensitivity on eligibility only (no nested schemes to save time)
        sens = []
        for b, m in [(120, 8), (150, 10), (200, 12)]:
            scfg = RankConfig(min_balls=b, min_matches=m, comp_scheme=COMP_SCHEME, n_bootstrap=60, seed=SEED)
            s = bootstrap_stable_ranks(match_agg, scfg)
            sens.append(
                {
                    "role": role,
                    "min_balls": b,
                    "min_matches": m,
                    "n_eligible": len(s),
                    "top10": " | ".join(s.head(10)["player_name"].tolist()) if len(s) else "",
                }
            )
            print(f"  sens {b}/{m}: " + (", ".join(s.head(5)["player_name"].tolist()) if len(s) else "none"))
        pd.DataFrame(sens).to_csv(OUTPUT_DIR / f"{role}_sensitivity_FINAL.csv", index=False)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    (OUTPUT_DIR / "final_submission_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"Done in {results['elapsed_sec']}s")
    return results


if __name__ == "__main__":
    run()
