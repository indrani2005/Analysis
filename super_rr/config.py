"""Locked modelling decisions from METHODOLOGY_PROPOSAL.md §H."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "6af458274c1e8_data_pack" / "Data Pack"
OUTPUT_DIR = ROOT / "outputs"
SEED = 42

# --- Match scope ---
EXCLUDE_SUPER_OVERS = True  # innings >= 3
PRIMARY_INNINGS = (1, 2)

# --- Attribution ---
ILLEGAL_EXTRAS = {"wide", "overstep"}  # case-insensitive
BOWLER_EXTRAS = {"wide", "overstep"}  # charged to bowler
NON_BOWLER_EXTRAS = {"byes", "legbyes", "penalty", "other"}
BOWLER_WICKET_TYPES = {
    "caught",
    "bowled",
    "lbw",
    "stump",
    "hit wicket",
}
# RunOut / Retire Out / Obstructing → not bowler wickets

# --- Pressure ---
PRESSURE_QUANTILE = 0.75  # high-pressure = top quartile of P_base (within scoped balls)
WP_LAMBDA = 0.25  # blend weight if WP leverage validates; else forced to 0
WP_MIN_AUC = 0.70  # match-blocked OOS gate for enabling leverage
WP_MAX_BRIER = 0.22

# Pressure base weights (innings 2)
W_RATE = 0.35
W_RES = 0.25
W_TIME = 0.25
W_CLOSE = 0.15

# Innings 1 weights
W1_PAR = 0.40
W1_WICKET = 0.30
W1_TIME = 0.30

# --- Expectations / roles ---
ROLE_BANDS = {
    "top": (1, 3),
    "middle": (4, 6),
    "lower": (7, 13),
}
PHASE_PP = (1, 6)
PHASE_MID = (7, 15)
PHASE_DEATH = (16, 20)

EB_PRIOR_BALLS = 40.0  # Empirical Bayes K for cell / player rates

# --- Eligibility (primary + sensitivity grid) ---
ELIGIBILITY_PRIMARY = (120, 8)  # pressure balls, distinct matches
ELIGIBILITY_GRID = [
    (80, 5),
    (120, 8),
    (150, 10),
    (200, 12),
]

# --- Ranking / bootstrap ---
N_BOOTSTRAP = 200  # primary stability draws (increase to 500–1000 for final freeze)
BOOTSTRAP_SEED = SEED
COMPOSITE_WEIGHTS = {
    "rae": 0.45,  # contextual residual (runs above / prevented)
    "clutch": 0.25,  # pressure residual − non-pressure residual
    "consistency": 0.30,  # share of pressure matches with positive residual
}

# Competition combine schemes to test
COMP_WEIGHT_SCHEMES = ("equal", "sample_proportional")
PRIMARY_COMP_SCHEME = "equal"

# IPL competition name marker
IPL_NAME_PATTERN = r"IPL"
