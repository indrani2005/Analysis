"""State reconstruction and pressure sanity checks."""

import numpy as np
import pandas as pd

from super_rr.pressure import pressure_base_innings2
from super_rr.state import reconstruct_state


def _mini_innings():
    rows = []
    # 6 legal balls: 1 run each, no wickets, innings 2 chase 120
    for i in range(1, 7):
        rows.append(
            {
                "match_id": 1,
                "innings": 2,
                "innings_ball_num": i,
                "overid": 1,
                "over_ball_num": i,
                "team_runs": 1.0,
                "is_wicket": False,
                "is_legal": True,
                "batting_order": 4,
                "revised_target": np.nan,
                "first_inn_revised_overs": np.nan,
                "second_inn_revised_overs": np.nan,
                "comp_family": "IPL",
            }
        )
    # Fake innings 1 total via a separate innings for target construction
    for i in range(1, 4):
        rows.append(
            {
                "match_id": 1,
                "innings": 1,
                "innings_ball_num": i,
                "overid": 1,
                "over_ball_num": i,
                "team_runs": 40.0 if i < 3 else 39.0,  # total 119
                "is_wicket": False,
                "is_legal": True,
                "batting_order": 1,
                "revised_target": np.nan,
                "first_inn_revised_overs": np.nan,
                "second_inn_revised_overs": np.nan,
                "comp_family": "IPL",
            }
        )
    return pd.DataFrame(rows)


def test_pre_delivery_no_lookahead():
    df = _mini_innings()
    out = reconstruct_state(df)
    inn2 = out[out["innings"] == 2].sort_values("innings_ball_num")
    assert inn2.iloc[0]["pre_runs"] == 0
    assert inn2.iloc[1]["pre_runs"] == 1
    assert inn2.iloc[0]["pre_legal_balls"] == 0
    assert inn2.iloc[5]["pre_legal_balls"] == 5


def test_high_rrr_late_more_pressure_than_easy_start():
    # Need 50 off 12 with 3 wickets vs need 50 off 60 with 8 wickets
    hard = pressure_base_innings2(
        rrr=[25.0], crr=[6.0], balls_remaining=[12], wickets_in_hand=[3], runs_needed=[50]
    )[0]
    easy = pressure_base_innings2(
        rrr=[5.0], crr=[8.0], balls_remaining=[60], wickets_in_hand=[8], runs_needed=[50]
    )[0]
    assert hard > easy
    assert hard > 0.5
    assert easy < 0.45
