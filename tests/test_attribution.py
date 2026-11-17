"""Unit tests for legal balls, extras, wickets, and player-ID joins."""

import pandas as pd
import pytest

from super_rr.attribution import (
    annotate_deliveries,
    bat_runs,
    bowler_runs_conceded,
    competition_family,
    find_name_collisions,
    is_bowler_wicket,
    is_legal_delivery,
    resolve_player_names,
    total_team_runs,
)


def test_legal_delivery_rules():
    assert is_legal_delivery(None) is True
    assert is_legal_delivery("wide") is False
    assert is_legal_delivery("Wide") is False
    assert is_legal_delivery("Overstep") is False
    assert is_legal_delivery("legbyes") is True
    assert is_legal_delivery("byes") is True


def test_bat_runs_off_the_bat_only():
    assert bat_runs(4) == 4
    assert bat_runs(0) == 0
    assert bat_runs(None) == 0


def test_team_runs_include_extras():
    assert total_team_runs(0, "wide", 1, 0) == 1
    assert total_team_runs(4, None, 0, 0) == 4
    assert total_team_runs(0, "legbyes", 4, 0) == 4


def test_bowler_not_charged_for_byes_legbyes():
    assert bowler_runs_conceded(0, "byes", 4, 0) == 0
    assert bowler_runs_conceded(0, "legbyes", 1, 0) == 0
    assert bowler_runs_conceded(0, "wide", 1, 0) == 1
    assert bowler_runs_conceded(4, "Overstep", 1, 0) == 5
    assert bowler_runs_conceded(6, None, 0, 0) == 6


def test_bowler_wicket_credit():
    assert is_bowler_wicket(True, "Caught") is True
    assert is_bowler_wicket(True, "Bowled") is True
    assert is_bowler_wicket(True, "LBW") is True
    assert is_bowler_wicket(True, "Stump") is True
    assert is_bowler_wicket(True, "Hit Wicket") is True
    assert is_bowler_wicket(True, "RunOut") is False
    assert is_bowler_wicket(True, "Retire Out") is False
    assert is_bowler_wicket(False, "Caught") is False


def test_annotate_deliveries_columns():
    df = pd.DataFrame(
        {
            "runs": [0, 4, 0, 1],
            "extras_type": ["wide", None, "legbyes", "Overstep"],
            "extras1_runs": [1, 0, 1, 1],
            "extras_2_runs": [0, 0, 0, 0],
            "is_wicket": [False, False, False, True],
            "dismissal_type": [None, None, None, "Bowled"],
            "dismissed_player_id": [None, None, None, 10],
            "striker_id": [1, 1, 1, 10],
            "is_four": [False, True, False, False],
            "is_six": [False, False, False, False],
        }
    )
    out = annotate_deliveries(df)
    assert list(out["is_legal"]) == [False, True, True, False]
    assert list(out["bowl_runs"]) == [1, 4, 0, 2]  # overstep: 1 bat + 1 extra? runs=1 → 1+1=2
    assert list(out["is_bowler_wicket"]) == [False, False, False, True]


def test_competition_family():
    assert competition_family("IPL 2025") == "IPL"
    assert competition_family("India in Australia T20I Series 2024") == "T20I"
    assert competition_family("Asia Cup Rising Stars 2025") == "T20I"


def test_player_id_name_collisions_not_merged():
    mapping = pd.DataFrame(
        {
            "comp_id": [1, 1, 2, 2],
            "comp_name": ["A", "A", "B", "B"],
            "team_id": [1, 1, 2, 2],
            "team_name": ["t", "t", "t", "t"],
            "player_id": [4722, 20172, 4722, 20172],
            "player_name": ["Rashid Khan", "Rashid Khan", "Rashid Khan", "Rashid Khan"],
        }
    )
    coll = find_name_collisions(mapping)
    assert "Rashid Khan" in set(coll["player_name"])
    canon = resolve_player_names(mapping)
    assert set(canon["player_id"]) == {4722, 20172}
    assert len(canon) == 2
