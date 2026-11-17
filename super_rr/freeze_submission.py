"""Freeze Unstop lists at 200 pressure balls / 12 matches after sensitivity."""

from __future__ import annotations

import json

import pandas as pd

from .config import OUTPUT_DIR, SEED
from .explain import build_explanation_card, write_explanation_cards
from .ratings import RankConfig, bootstrap_stable_ranks, top10_frequency

MIN_BALLS, MIN_MATCHES, N_BOOT = 200, 12, 100


def freeze(role: str) -> pd.DataFrame:
    match_agg = pd.read_csv(OUTPUT_DIR / f"{role}_match_aggregates_final.csv")
    df = pd.read_csv(OUTPUT_DIR / "deliveries_modeled.csv.gz", low_memory=False)
    cfg = RankConfig(
        min_balls=MIN_BALLS,
        min_matches=MIN_MATCHES,
        comp_scheme="equal",
        n_bootstrap=N_BOOT,
        seed=SEED,
    )
    ranked = bootstrap_stable_ranks(match_agg, cfg)
    cand = ranked.head(25)["player_id"].astype(int).tolist()
    freq = top10_frequency(match_agg, cfg, cand)
    ranked["top10_freq"] = ranked["player_id"].astype(int).map(freq).fillna(0.0)
    ranked.to_csv(OUTPUT_DIR / f"{role}_rankings_SUBMIT.csv", index=False)
    top10 = ranked.head(10).copy()
    top10.to_csv(OUTPUT_DIR / f"{role}_top10_SUBMIT.csv", index=False)
    cards = [build_explanation_card(r, df, role) for _, r in top10.iterrows()]
    write_explanation_cards(cards, OUTPUT_DIR / f"{role}_explanation_cards_SUBMIT.json")
    print(f"\n=== {role.upper()} TOP 10 (SUBMIT) ===")
    for _, r in top10.iterrows():
        print(
            f"{int(r['rank']):2d}. {r['player_name']} | id={int(r['player_id'])} | "
            f"R*={r['R_stable']:.3f} | balls={r['pressure_balls']:.0f} | "
            f"matches={r['pressure_matches']:.0f} | top10%={r['top10_freq']:.0%} | {r['families']}"
        )
    return top10


def main():
    out = {"eligibility": f"{MIN_BALLS}/{MIN_MATCHES}", "n_bootstrap": N_BOOT}
    for role in ("batter", "bowler"):
        t = freeze(role)
        out[role] = t[
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
    (OUTPUT_DIR / "UNSTOP_SUBMISSION.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Human dropdown sheet
    lines = [
        "# Unstop dropdown — FINAL SUBMISSION",
        "",
        f"Eligibility freeze: **{MIN_BALLS} pressure balls + {MIN_MATCHES} matches**, role filter on, equal IPL/T20I combine, rank by R_stable = boot_mean − boot_sd.",
        "",
        "## Top 10 Batters",
    ]
    for r in out["batter"]:
        lines.append(
            f"{r['rank']}. **{r['player_name']}** (player_id `{int(r['player_id'])}`)"
        )
    lines += ["", "## Top 10 Bowlers"]
    for r in out["bowler"]:
        lines.append(
            f"{r['rank']}. **{r['player_name']}** (player_id `{int(r['player_id'])}`)"
        )
    (OUTPUT_DIR / "UNSTOP_DROPDOWNS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote outputs/UNSTOP_DROPDOWNS.md and UNSTOP_SUBMISSION.json")


if __name__ == "__main__":
    main()
