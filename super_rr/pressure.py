"""Transparent resource/time/rate pressure + optional WP leverage blend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    PRESSURE_QUANTILE,
    W_CLOSE,
    W_RATE,
    W_RES,
    W_TIME,
    W1_PAR,
    W1_TIME,
    W1_WICKET,
    WP_LAMBDA,
)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pressure_base_innings2(rrr, crr, balls_remaining, wickets_in_hand, runs_needed) -> np.ndarray:
    rrr = np.asarray(rrr, dtype=float)
    crr = np.asarray(crr, dtype=float)
    br = np.asarray(balls_remaining, dtype=float)
    wih = np.asarray(wickets_in_hand, dtype=float)
    rn = np.asarray(runs_needed, dtype=float)

    gap = np.maximum(0.0, rrr - crr)
    # Rate stress: gap + absolute RRR
    s_rate = _clip01(0.55 * _sigmoid((gap - 1.5) / 1.2) + 0.45 * _sigmoid((rrr - 9.0) / 2.0))

    # Resource stress: few wickets and/or few balls
    # wickets_lost_ratio style: low wickets_in_hand → high stress
    wicket_scarce = _clip01((6.0 - wih) / 6.0)
    ball_scarce = _clip01((36.0 - br) / 36.0)
    s_res = _clip01(0.6 * wicket_scarce + 0.4 * ball_scarce)

    # Time / deadline stress
    s_time = _clip01((48.0 - br) / 48.0)

    # Closeness: small chases late (decision boundary)
    close = ((rn <= 20) & (br <= 18)).astype(float)
    tight = ((np.abs(rrr - 8.5) <= 2.0) & (br <= 30) & (wih <= 5)).astype(float)
    s_close = _clip01(0.7 * close + 0.3 * tight)

    return _clip01(
        W_RATE * s_rate + W_RES * s_res + W_TIME * s_time + W_CLOSE * s_close
    )


def pressure_base_innings1(par_deficit, wickets_lost, balls_remaining, crr) -> np.ndarray:
    pd_ = np.asarray(par_deficit, dtype=float)
    wl = np.asarray(wickets_lost, dtype=float)
    br = np.asarray(balls_remaining, dtype=float)
    crr = np.asarray(crr, dtype=float)

    # Behind par → batting pressure; ahead with wickets in hand late → bowling pressure
    # We store a signed state pressure for the batting side; bowling uses same P as situation difficulty.
    s_par = _clip01(_sigmoid(pd_ / 15.0))  # deficit positive → higher bat pressure
    # Collapse stress: wickets lost early
    frac_gone = wl / 9.0
    early = br / 120.0
    s_wkt = _clip01(frac_gone * (0.5 + 0.5 * early) / 0.7)

    # Late innings intensity (death overs obligation / defending)
    s_time = _clip01((36.0 - br) / 36.0)
    # Inflate if scoring slowly late OR scoring very fast with wickets (bowling defence pressure)
    death_bat = s_time * _clip01(_sigmoid((7.5 - crr) / 1.5))
    death_bowl = s_time * _clip01(_sigmoid((crr - 9.5) / 1.5)) * _clip01((10 - wl) / 6.0)
    s_time_eff = _clip01(np.maximum(death_bat, death_bowl) * 0.7 + s_time * 0.3)

    return _clip01(W1_PAR * s_par + W1_WICKET * s_wkt + W1_TIME * s_time_eff)


def compute_pressure_base(df: pd.DataFrame) -> pd.Series:
    p = np.zeros(len(df), dtype=float)
    m2 = df["innings"].to_numpy() == 2
    m1 = df["innings"].to_numpy() == 1
    if m2.any():
        p[m2] = pressure_base_innings2(
            df.loc[m2, "rrr"].to_numpy(),
            df.loc[m2, "crr"].to_numpy(),
            df.loc[m2, "balls_remaining"].to_numpy(),
            df.loc[m2, "wickets_in_hand"].to_numpy(),
            df.loc[m2, "runs_needed"].to_numpy(),
        )
    if m1.any():
        p[m1] = pressure_base_innings1(
            df.loc[m1, "par_deficit"].to_numpy(),
            df.loc[m1, "pre_wkts"].to_numpy(),
            df.loc[m1, "balls_remaining"].to_numpy(),
            df.loc[m1, "crr"].to_numpy(),
        )
    return pd.Series(p, index=df.index, name="P_base")


def blend_pressure(p_base: pd.Series, p_lev: pd.Series | None, lambda_: float) -> pd.Series:
    if p_lev is None or lambda_ <= 0:
        return p_base.rename("P")
    p = (1.0 - lambda_) * p_base.to_numpy() + lambda_ * p_lev.to_numpy()
    return pd.Series(_clip01(p), index=p_base.index, name="P")


def flag_high_pressure(p: pd.Series, quantile: float = PRESSURE_QUANTILE) -> tuple[pd.Series, float]:
    thr = float(p.quantile(quantile))
    return (p >= thr).rename("is_pressure"), thr


def intensity_weight(p: pd.Series, thr: float) -> pd.Series:
    """Continuous weight: 0 below thr, rises with how far above thr."""
    w = np.clip((p.to_numpy() - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
    # Keep a minimum weight of 1 for pressure balls for sample counting; intensity for scoring
    return pd.Series(w, index=p.index, name="pressure_intensity")
