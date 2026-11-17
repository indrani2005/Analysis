"""Match-blocked out-of-fold win probability and leverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import SEED, WP_LAMBDA, WP_MAX_BRIER, WP_MIN_AUC


def _match_winner_batting(df: pd.DataFrame) -> pd.Series:
    """
    Binary label: does the current batting side eventually win the match?
    """
    win = df["win_team_id"]
    bat = df["batting_team_id"]
    # If no winner (should already be excluded), mark NaN
    y = (win == bat).astype(float)
    y = y.where(win.notna(), np.nan)
    return y


def build_wp_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["innings"] = df["innings"].astype(float)
    X["balls_remaining"] = df["balls_remaining"].astype(float)
    X["wickets_in_hand"] = df["wickets_in_hand"].astype(float)
    X["pre_runs"] = df["pre_runs"].astype(float)
    X["crr"] = df["crr"].fillna(0).astype(float)
    X["rrr"] = df["rrr"].fillna(0).astype(float)
    X["runs_needed"] = df["runs_needed"].fillna(0).astype(float)
    X["is_inn2"] = (df["innings"] == 2).astype(float)
    X["par_deficit"] = df["par_deficit"].fillna(0).astype(float)
    X["knock_out"] = df["knock_out"].fillna(0).astype(float)
    return X


def fit_oof_win_probability(
    df: pd.DataFrame, n_splits: int = 5, seed: int = SEED
) -> tuple[pd.Series, dict]:
    """
    Match-grouped OOF logistic WP for batting side win.

    Returns OOF predicted WP aligned to df.index and a metrics dict.
    If gates fail, caller should set lambda=0.
    """
    y = _match_winner_batting(df)
    usable = y.notna()
    X_all = build_wp_features(df)
    wp = pd.Series(np.nan, index=df.index, name="wp_bat")

    metrics = {"enabled": False, "auc": None, "brier": None, "n": int(usable.sum()), "lambda": 0.0}

    if usable.sum() < 5000 or df.loc[usable, "match_id"].nunique() < 50:
        return wp, metrics

    X = X_all.loc[usable]
    yy = y.loc[usable].astype(int)
    groups = df.loc[usable, "match_id"]

    n_groups = groups.nunique()
    splits = min(n_splits, max(2, n_groups // 20))
    gkf = GroupKFold(n_splits=splits)
    oof = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in gkf.split(X, yy, groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X.iloc[train_idx])
        Xte = scaler.transform(X.iloc[test_idx])
        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(Xtr, yy.iloc[train_idx])
        oof[test_idx] = clf.predict_proba(Xte)[:, 1]

    wp.loc[usable] = oof
    try:
        auc = float(roc_auc_score(yy, oof))
    except ValueError:
        auc = float("nan")
    brier = float(brier_score_loss(yy, oof))
    metrics.update({"auc": auc, "brier": brier})

    if (auc >= WP_MIN_AUC) and (brier <= WP_MAX_BRIER):
        metrics["enabled"] = True
        metrics["lambda"] = WP_LAMBDA
    else:
        metrics["enabled"] = False
        metrics["lambda"] = 0.0
    return wp, metrics


def leverage_from_wp(df: pd.DataFrame, wp: pd.Series) -> pd.Series:
    """
    Approximate pre-delivery leverage without outcome look-ahead.

    Uses state-conditioned expected absolute WP swing under a simple
    outcome distribution proxy: higher when WP near 0.5 and few resources left.
    """
    p = wp.fillna(0.5).to_numpy()
    # Uncertainty peak near 0.5
    balance = 1.0 - np.abs(p - 0.5) * 2.0
    br = df["balls_remaining"].to_numpy(dtype=float)
    late = np.clip((48.0 - br) / 48.0, 0.0, 1.0)
    # Chase tightness boost
    inn2 = (df["innings"].to_numpy() == 2).astype(float)
    rrr = df["rrr"].fillna(0).to_numpy(dtype=float)
    tight = inn2 * np.clip(np.abs(rrr - 8.5) / 6.0, 0.0, 1.0)
    lev = np.clip(0.50 * balance + 0.30 * late + 0.20 * tight, 0.0, 1.0)
    # If WP missing, fall back to late*tight only
    missing = wp.isna().to_numpy()
    lev[missing] = np.clip(0.6 * late[missing] + 0.4 * tight[missing], 0.0, 1.0)
    return pd.Series(lev, index=df.index, name="P_lev")
