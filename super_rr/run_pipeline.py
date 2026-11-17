"""
End-to-end SupeRR Selector pipeline.

Usage (from repo root):
  python -m super_rr.run_pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import (
    COMP_WEIGHT_SCHEMES,
    ELIGIBILITY_GRID,
    ELIGIBILITY_PRIMARY,
    N_BOOTSTRAP,
    OUTPUT_DIR,
    PRIMARY_COMP_SCHEME,
    SEED,
)
from .explain import build_explanation_card, write_explanation_cards
from .expectations import compute_expected_outcomes
from .load_data import build_delivery_table, save_audit
from .pressure import (
    blend_pressure,
    compute_pressure_base,
    flag_high_pressure,
    intensity_weight,
)
from .ratings import (
    RankConfig,
    bootstrap_stable_ranks,
    player_match_aggregates,
    top10_frequency,
)
from .state import attach_par, reconstruct_state
from .win_prob import fit_oof_win_probability, leverage_from_wp


def run(n_bootstrap: int | None = None) -> dict:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_boot = n_bootstrap if n_bootstrap is not None else N_BOOTSTRAP

    print("==> Loading & scoping deliveries")
    scoped, audit = build_delivery_table()
    save_audit(audit)

    print("==> Reconstructing pre-delivery state")
    state = reconstruct_state(scoped)
    state = attach_par(state)

    print("==> Base pressure engine")
    state["P_base"] = compute_pressure_base(state)

    print("==> Match-blocked win probability (conditional leverage)")
    wp, wp_metrics = fit_oof_win_probability(state)
    state["wp_bat"] = wp
    state["P_lev"] = leverage_from_wp(state, wp)
    lam = float(wp_metrics["lambda"])
    state["P"] = blend_pressure(state["P_base"], state["P_lev"], lam)
    is_p, thr = flag_high_pressure(state["P"])
    state["is_pressure"] = is_p
    state["pressure_intensity"] = intensity_weight(state["P"], thr)
    audit["pressure_threshold"] = thr
    audit["pressure_balls"] = int(state["is_pressure"].sum())
    audit["wp_metrics"] = wp_metrics
    save_audit(audit)
    print(
        f"    WP enabled={wp_metrics['enabled']} AUC={wp_metrics['auc']} "
        f"Brier={wp_metrics['brier']} lambda={lam} | P thr={thr:.3f} "
        f"pressure_balls={audit['pressure_balls']}"
    )

    print("==> Contextual expectations + residuals")
    modeled = compute_expected_outcomes(state)

    # Persist delivery-level sample for audit (parquet if possible, else csv.gz)
    deliv_path = OUTPUT_DIR / "deliveries_modeled.csv.gz"
    keep_cols = [
        c
        for c in [
            "match_id",
            "innings",
            "innings_ball_num",
            "comp_family",
            "striker_id",
            "striker_name_canon",
            "bowler_id",
            "bowler_name_canon",
            "is_legal",
            "bat_runs",
            "bowl_runs",
            "is_bowler_wicket",
            "is_striker_out",
            "pre_runs",
            "pre_wkts",
            "balls_remaining",
            "crr",
            "rrr",
            "target",
            "phase",
            "role_band",
            "P_base",
            "P_lev",
            "P",
            "is_pressure",
            "pressure_intensity",
            "exp_bat_runs",
            "exp_bowl_runs",
            "bat_residual",
            "bowl_residual",
            "bat_cell",
            "bowl_cell",
        ]
        if c in modeled.columns
    ]
    modeled[keep_cols].to_csv(deliv_path, index=False, compression="gzip")

    results = {"audit": audit, "wp_metrics": wp_metrics, "elapsed_sec": None}

    for role in ("batter", "bowler"):
        print(f"==> Rating {role}s")
        match_agg = player_match_aggregates(modeled, role)
        match_agg.to_csv(OUTPUT_DIR / f"{role}_match_aggregates.csv", index=False)

        cfg = RankConfig(
            min_balls=ELIGIBILITY_PRIMARY[0],
            min_matches=ELIGIBILITY_PRIMARY[1],
            comp_scheme=PRIMARY_COMP_SCHEME,
            n_bootstrap=n_boot,
            seed=SEED,
        )
        ranked = bootstrap_stable_ranks(match_agg, cfg)
        if ranked.empty:
            print(f"    WARNING: no eligible {role}s")
            results[role] = {"top10": []}
            continue

        # Top-10 frequency for shortlisted pool (top 25 candidates by R_stable)
        cand = ranked.head(25)["player_id"].astype(int).tolist()
        freq = top10_frequency(match_agg, cfg, cand)
        ranked["top10_freq"] = ranked["player_id"].astype(int).map(freq).fillna(0.0)

        ranked.to_csv(OUTPUT_DIR / f"{role}_rankings_full.csv", index=False)
        top10 = ranked.head(10).copy()
        top10.to_csv(OUTPUT_DIR / f"{role}_top10.csv", index=False)

        cards = [build_explanation_card(r, modeled, role) for _, r in top10.iterrows()]
        write_explanation_cards(cards, OUTPUT_DIR / f"{role}_explanation_cards.json")

        print("    TOP 10:")
        for _, r in top10.iterrows():
            print(
                f"      {int(r['rank']):2d}. {r['player_name']} "
                f"(id={int(r['player_id'])}) R_stable={r['R_stable']:.3f} "
                f"balls={r['pressure_balls']:.0f} top10%={r['top10_freq']:.0%}"
            )
        results[role] = {
            "top10": top10[
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
                ]
            ].to_dict(orient="records")
        }

        # Sensitivity: eligibility grid + competition schemes
        sens_rows = []
        for balls, matches in ELIGIBILITY_GRID:
            for scheme in COMP_WEIGHT_SCHEMES:
                scfg = RankConfig(
                    min_balls=balls,
                    min_matches=matches,
                    comp_scheme=scheme,
                    n_bootstrap=max(100, n_boot // 5),
                    seed=SEED,
                )
                srank = bootstrap_stable_ranks(match_agg, scfg)
                names = (
                    srank.head(10)["player_name"].tolist() if not srank.empty else []
                )
                sens_rows.append(
                    {
                        "role": role,
                        "min_balls": balls,
                        "min_matches": matches,
                        "comp_scheme": scheme,
                        "n_eligible": int(len(srank)),
                        "top10": " | ".join(names),
                    }
                )
                print(
                    f"    sensitivity {balls}/{matches} {scheme}: "
                    f"n={len(srank)} | " + ", ".join(names[:5]) + ("..." if names else "")
                )
        pd.DataFrame(sens_rows).to_csv(
            OUTPUT_DIR / f"{role}_sensitivity.csv", index=False
        )

    results["elapsed_sec"] = round(time.time() - t0, 1)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"==> Done in {results['elapsed_sec']}s → {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    run()
