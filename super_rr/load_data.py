"""Load and scope the official Data Pack."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .attribution import annotate_deliveries, competition_family, resolve_player_names
from .config import DATA_DIR, EXCLUDE_SUPER_OVERS, OUTPUT_DIR, PRIMARY_INNINGS


def load_raw(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    d = Path(data_dir) if data_dir else DATA_DIR
    return {
        "bbb": pd.read_csv(d / "ball_by_ball.csv", low_memory=False),
        "matches": pd.read_csv(d / "matches.csv"),
        "competitions": pd.read_csv(d / "competitions.csv"),
        "teams": pd.read_csv(d / "teams.csv"),
        "playing_eleven": pd.read_csv(d / "playing_eleven.csv"),
        "mapping": pd.read_csv(d / "competition_team_player_mapping.csv"),
    }


def _is_abandoned_or_no_result(result) -> bool:
    if pd.isna(result):
        return False
    s = str(result).strip().lower()
    return ("abandon" in s) or (s == "no result") or s.startswith("no result")


def scope_matches(matches: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    m = matches.merge(
        competitions[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
    )
    m["comp_family"] = m["comp_name"].map(competition_family)
    m["exclude_result"] = m["result"].map(_is_abandoned_or_no_result)
    m["in_primary_scope"] = ~m["exclude_result"]
    return m


def build_delivery_table(raw: dict[str, pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Return annotated, scoped delivery table + audit dict.

    Exclusions applied here:
    - matches with abandoned / no-result
    - super overs (innings >= 3) when EXCLUDE_SUPER_OVERS
    - matches with no ball-by-ball rows (already absent from bbb)
    """
    raw = raw or load_raw()
    bbb = annotate_deliveries(raw["bbb"])
    matches = scope_matches(raw["matches"], raw["competitions"])
    players = resolve_player_names(raw["mapping"])

    # Join match meta
    meta_cols = [
        "match_id",
        "comp_id",
        "comp_name",
        "match_format",
        "season",
        "comp_family",
        "venue_id",
        "knock_out",
        "stage",
        "revised_target",
        "first_inn_revised_overs",
        "second_inn_revised_overs",
        "win_team_id",
        "result",
        "exclude_result",
        "in_primary_scope",
        "home_team_id",
        "away_team_id",
        "toss_win_team_id",
        "toss_opted",
    ]
    df = bbb.merge(matches[meta_cols], on="match_id", how="left")

    audit = {
        "raw_deliveries": int(len(raw["bbb"])),
        "raw_matches": int(len(raw["matches"])),
        "bbb_matches": int(raw["bbb"]["match_id"].nunique()),
        "abandoned_or_no_result_matches": int(matches["exclude_result"].sum()),
        "super_over_deliveries": int((df["innings"] >= 3).sum()),
    }

    # Scope filter
    mask = df["in_primary_scope"].fillna(False)
    if EXCLUDE_SUPER_OVERS:
        mask &= df["innings"].isin(PRIMARY_INNINGS)
    scoped = df.loc[mask].copy()
    scoped = scoped.sort_values(
        ["match_id", "innings", "innings_ball_num"]
    ).reset_index(drop=True)

    # Attach canonical names
    scoped = scoped.merge(
        players.rename(columns={"player_id": "striker_id", "player_name_canon": "striker_name_canon"}),
        on="striker_id",
        how="left",
    )
    scoped = scoped.merge(
        players.rename(columns={"player_id": "bowler_id", "player_name_canon": "bowler_name_canon"}),
        on="bowler_id",
        how="left",
    )
    scoped["striker_name_canon"] = scoped["striker_name_canon"].fillna(scoped["striker_name"])
    scoped["bowler_name_canon"] = scoped["bowler_name_canon"].fillna(scoped["bowler_name"])

    audit.update(
        {
            "scoped_deliveries": int(len(scoped)),
            "scoped_matches": int(scoped["match_id"].nunique()),
            "scoped_legal": int(scoped["is_legal"].sum()),
            "ipl_matches": int(scoped.loc[scoped["comp_family"] == "IPL", "match_id"].nunique()),
            "t20i_matches": int(scoped.loc[scoped["comp_family"] == "T20I", "match_id"].nunique()),
            "revised_target_matches_in_scope": int(
                scoped.loc[scoped["revised_target"].notna(), "match_id"].nunique()
            ),
        }
    )
    return scoped, audit


def save_audit(audit: dict, path: Path | None = None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (OUTPUT_DIR / "data_quality_audit.json")
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path
