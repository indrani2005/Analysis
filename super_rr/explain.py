"""Per-player explanation cards for shortlisted rankings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def confidence_label(top10_freq: float) -> str:
    if top10_freq >= 0.90:
        return "Lock"
    if top10_freq >= 0.60:
        return "Strong"
    return "Provisional"


def build_explanation_card(
    row: pd.Series,
    delivery_df: pd.DataFrame,
    role: str,
) -> dict:
    pid = int(row["player_id"])
    if role == "batter":
        sub = delivery_df[
            (delivery_df["striker_id"] == pid) & (delivery_df["is_legal"])
        ]
        resid_col = "bat_residual"
        runs_col = "bat_runs"
        exp_col = "exp_bat_runs"
    else:
        sub = delivery_df[delivery_df["bowler_id"] == pid]
        resid_col = "bowl_residual"
        runs_col = "bowl_runs"
        exp_col = "exp_bowl_runs"

    press = sub[sub["is_pressure"]]
    non = sub[~sub["is_pressure"]]
    actual = float(press[runs_col].sum()) if len(press) else 0.0
    expected = float(press[exp_col].sum()) if len(press) else 0.0
    # Batters: positive = scored more than expected.
    # Bowlers: positive = prevented runs (expected conceded − actual conceded).
    value_vs_exp = (actual - expected) if role == "batter" else (expected - actual)
    match_pos = (
        press.groupby("match_id")[resid_col].mean() > 0
    ).mean() if len(press) else 0.0

    card = {
        "rank": int(row.get("rank", -1)),
        "player_id": pid,
        "player_name": row.get("player_name"),
        "role": role,
        "families": row.get("families"),
        "pressure_balls": float(row.get("pressure_balls", len(press))),
        "pressure_matches": float(row.get("pressure_matches", press["match_id"].nunique() if len(press) else 0)),
        "actual_pressure_runs": actual,
        "expected_pressure_runs": expected,
        "runs_above_expected": value_vs_exp,
        "pressure_residual_mean": float(press[resid_col].mean()) if len(press) else None,
        "nonpressure_residual_mean": float(non[resid_col].mean()) if len(non) else None,
        "clutch_delta": float(row.get("clutch", np.nan)),
        "consistency": float(row.get("consistency", match_pos)),
        "R_mean": float(row.get("R_shrunk", row.get("R_mean", np.nan))),
        "boot_mean": float(row.get("boot_mean", np.nan)),
        "boot_sd": float(row.get("boot_sd", np.nan)),
        "R_stable": float(row.get("R_stable", np.nan)),
        "R_lcb90": float(row.get("R_lcb90", np.nan)),
        "top10_freq": float(row.get("top10_freq", np.nan)),
        "confidence": confidence_label(float(row.get("top10_freq", 0.0) or 0.0)),
        "why": _why_text(row, value_vs_exp, role),
    }
    return card


def _why_text(row: pd.Series, value_vs_exp: float, role: str) -> str:
    verb = (
        "outperformed situational expectation"
        if value_vs_exp >= 0
        else "underperformed situational expectation"
    )
    unit = (
        "runs above expected"
        if role == "batter"
        else "runs prevented vs expected"
    )
    return (
        f"{row.get('player_name')} {verb} under pressure ({unit}: {value_vs_exp:+.1f}). "
        f"Consistency={float(row.get('consistency', 0)):.0%} of pressure matches above residual zero; "
        f"bootstrap stability R_stable={float(row.get('R_stable', np.nan)):.3f} "
        f"(mean={float(row.get('boot_mean', np.nan)):.3f}, sd={float(row.get('boot_sd', np.nan)):.3f}). "
        f"Top-10 frequency={float(row.get('top10_freq', 0) or 0):.0%} → {confidence_label(float(row.get('top10_freq', 0) or 0))}."
    )


def write_explanation_cards(cards: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    # Also markdown
    md_path = path.with_suffix(".md")
    lines = ["# Player Explanation Cards\n"]
    for c in cards:
        lines.append(f"## Rank {c['rank']}: {c['player_name']} (`{c['player_id']}`)\n")
        lines.append(f"- Role: {c['role']}")
        lines.append(f"- Competitions: {c['families']}")
        lines.append(f"- Pressure sample: {c['pressure_balls']:.0f} balls / {c['pressure_matches']:.0f} matches")
        lines.append(
            f"- Actual vs expected pressure runs: {c['actual_pressure_runs']:.1f} vs {c['expected_pressure_runs']:.1f} "
            f"(Δ {c['runs_above_expected']:+.1f})"
        )
        lines.append(f"- Consistency: {c['consistency']:.0%}")
        lines.append(
            f"- Ranking uncertainty: R_stable={c['R_stable']:.3f}, boot_mean={c['boot_mean']:.3f}, "
            f"boot_sd={c['boot_sd']:.3f}, LCB90={c['R_lcb90']:.3f}"
        )
        lines.append(f"- Confidence: **{c['confidence']}** (top-10 freq {c['top10_freq']:.0%})")
        lines.append(f"- Why: {c['why']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
