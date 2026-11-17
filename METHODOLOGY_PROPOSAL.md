# SupeRR Selector Season 2 — Research Brief & Methodology Proposal

**Status:** APPROVED (with modifications) — implementation authorized  
**Dataset:** Official Data Pack (`6af458274c1e8_data_pack`)  
**Core question:** When a player faces a genuinely difficult situation, do they outperform what would normally be expected *in that situation* — and can they repeat it?

---

## A. Dataset Schema & Quality Audit

### Files & sizes

| File | Rows | Role |
|---|---:|---|
| `ball_by_ball.csv` | 200,999 | Delivery-level events (33 columns) |
| `matches.csv` | 883 | Match metadata, revisions, results |
| `competitions.csv` | 141 | Competition / season / format |
| `teams.csv` | 353 | Competition-scoped teams |
| `playing_eleven.csv` | 28,381 | XI, batting order, impact flags |
| `competition_team_player_mapping.csv` | 6,413 | Official player IDs/names for Unstop |

### Coverage

- **Matches with ball data:** 867 of 883 (16 abandoned matches have no deliveries — all `Match Abandoned`)
- **Format split (matches table):** T20I 482 · T20 401
- **IPL:** 371 matches across IPL 2022–2026 (`comp_id` 1358, 1715, 2193, 2725, 3298)
- **Seasons present:** 2022 → 2026 (includes 2022/23, 2023/24, 2024/25 labels)
- **Non-IPL T20 (format=T20):** Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025 (plus IPL)
- **Super overs:** innings 3/4 exist (38 deliveries)
- **Knockouts:** 31 matches flagged `knock_out=1`
- **Revised/DLS-like:** 23 revised targets; 25 matches with revised overs

### Ball-by-ball schema (key fields)

`match_id, innings, overid, over_ball_num, innings_ball_num, batting_team_id, bowling_team_id, striker_id/name, non_striker_id/name, bowler_id/name, batting_type, bowling_type, batting_order, runs, is_four, is_six, is_wicket, dismissal_type(_id), dismissed_player_id, extras_type, extras1_runs, extras_2_runs, pace_or_spin, bowl_speed, stroke, delivery_style, over_or_around, on_or_off, delivery_ist_time`

### Cleaning decisions (proposed)

1. **Legal deliveries:** treat as legal if `extras_type` ∉ {`wide`, `Overstep`}. Approx **192,709** legal balls.
2. **Bat runs:** `runs` appears to be **off-the-bat only** (wides/byes have `runs=0`; extras live in `extras1_runs`). Use `runs` for batting SR; for bowling economy use bat runs + wides/no-balls charged to bowler, **exclude byes/leg-byes**.
3. **Bowler wickets:** credit only Caught / Bowled / LBW / Stump / Hit Wicket. **Do not** credit RunOut / Retire Out / Obstructing.
4. **Duplicates:** none on `(match_id, innings, innings_ball_num)` — good.
5. **Player IDs:** map file has **7 name collisions** (same display name, different IDs), including two `Rashid Khan` IDs (`4722`, `20172`). Rankings must use `player_id`, never name alone.
6. **Incomplete contests:** exclude abandoned and no-result matches from final rankings and WP training. Retain revised-target matches only when overs/target/state can be reconstructed accurately. **Exclude super overs (innings ≥ 3) from primary rankings** (optional separate appendix).
7. **Do not invent data.** Document every exclusion with counts.

### Important audit note

Public projects report ~192k–201k deliveries and ~805–883 matches. Our pack is **200,999 / 883** raw; usable ball matches = **867**. Any copied ranking from another repo is invalid until reproduced on *this* pack.

---

## B. Comparative Analysis of Existing Methodologies

| Project | Pressure definition | Performance vs expectation | Reliability | Distinctive move |
|---|---|---|---|---|
| **Tejesh Chowdary** | Continuous PI from match closeness, over, wickets, run stress, tournament; chase multiplier | RAE / Runs Saved + clutch Δ vs low-pressure | EB shrinkage (α≈60), consistency %, ≥200 HP balls | 4-pillar PRI; polished dashboard |
| **Aniket Kumar** | CPI (RRR/wickets/phase/closeness) + WP leverage hybrid (AUC≈0.89) | ΔSR / ΔEcon + Press WPA | EB (K=40), ≥45 pressure balls / ≥8 matches | End-to-end WP + simulator |
| **Ishaan Kumar** | Top 25% win-prob leverage (OOS-tested WP) | Output vs **situation par**, then vs player’s usual vs-par | 400 match redraws; Lock/Strong/Selected labels | Rank **stability** as first-class output |
| **Sughan** | WP-calibrated, per-innings top 30% pressure | Unified effectiveness `p_eff` with IPL/non-IPL EB | **1,000 match-cluster bootstraps**; rank by **1-SD lower bound** | Uncertainty-aware official ranking |
| **Chauhan Aman** | WP leverage + tension + stress + stage | Context-adjusted delivery credit + opponent LOMO | EB + **H_low = posterior − 1.645 SE**; validation gates | Honest: labels lists **EXPLORATORY** because gates fail |
| **theic-alt** | Binary/rule pressure (RRR, wickets, balls rem; separate inn1 bat/bowl rules) | z-scored SR/econ + uplifts | Reliability = n/(n+60); competition weights | Role-split inn1 pressure rules; UI-heavy |
| **Shivang Clutch Index** *(related, not SupeRR)* | Chase-only continuous 0–1 pressure | High-pressure tertile Δ vs own conventional | Shrink toward 0; shared tertiles | Clean auditability; exposes SR-vs-average trap |

### Ranking disagreement (proof that method > reputation)

Across public top-10s alone: SKY / Rinku / Stubbs / Tilak / Tim David recur for batting, but Sughan elevates Sooryavanshi/Jaiswal/Kamindu; theic elevates Ferreira/Ashutosh/Raza. Bowling: Pathirana/Bumrah/Narine recur, but Umran / Hazlewood / Muqeem / Abrar appear depending on WP vs economy framing. **Do not copy any list.**

---

## C. Weaknesses & Differentiation Opportunities

### Common weaknesses

1. **Death-over / late-chase heuristics** still dominate weaker entries; continuous pressure is claimed but sometimes poorly calibrated.
2. **First-innings pressure** is often shallow (death overs + wickets) vs rich chase equations.
3. **Expectation baselines** often global or phase-only — weak role/context matching (finisher ≠ opener).
4. **Arbitrary weights** (0.35/0.25/…) rarely sensitivity-tested to stability.
5. **Sample thresholds** differ wildly (45 vs 150–200 balls) → different shortlists.
6. **WP models** can be circular if pressure and credit both use the same in-sample WP without match-blocked OOS validation (Chauhan’s failed gates are a warning).
7. **Name-based aggregation risk** given ID collisions in the official map.
8. **Complexity theater:** dashboards ≠ statistical reliability.

### Our differentiators (intentional)

1. **Contextual expected performance** by situation *and* batting role/order band (not only “pressure SR”).
2. **Shared pressure engine, separate bat/bowl outcome models.**
3. **Intensity-weighted** credit (not all pressure balls equal).
4. **Reliability-first ranking:** EB + match-consistency + bootstrap **lower-bound** score.
5. **Mandatory sensitivity & stability report** (Ishaan/Sughan/Chauhan lessons).
6. **Strict credit accounting** for extras/wickets (often glossed over).

---

## D. Proposed Original Pressure Model

### Design principles

- Use **pre-delivery** state only (no look-ahead).
- Continuous score \(P \in [0,1]\), not “overs 16–20 only”.
- Separate formulas for **Innings 1** and **Innings 2** (and ignore/super-over handle separately).
- Optional leverage term only if WP validates **out-of-fold by match**.

### State reconstruction (before each ball)

Rebuild: runs scored, wickets lost, legal balls bowled/remaining, CRR, target (inn2 / revised), RRR, phase (PP 1–6, middle 7–15, death 16–20), resources remaining.

### Innings 2 — Chase Pressure

Components (normalized to [0,1]):

1. **Rate stress** \(S_{rate}\): based on \(\max(0, RRR - CRR)\) and absolute RRR (gap matters more than raw RRR).
2. **Resource stress** \(S_{res}\): wickets in hand × balls remaining (low resources → high stress).
3. **Deadline stress** \(S_{time}\): balls remaining (nonlinear; sharper in last 30–36 balls).
4. **Closeness / leverage** \(S_{lev}\): from OOS WP swing *or* a transparent proxy (runs needed & balls left near decision boundary).

$$P_2 = \mathrm{clip}\big(w_1 S_{rate} + w_2 S_{res} + w_3 S_{time} + w_4 S_{lev}\big)$$

Initial weight proposal (to be sensitivity-tested): \(0.35, 0.25, 0.20, 0.20\).

### Innings 1 — Setting Pressure

No target → use **par-relative tension**:

1. Projected finish vs venue/competition **par** (historical innings projection from resources).
2. Wicket collapse stress early/middle (losing 4+ with many overs left).
3. Death-overs scoring obligation when behind par / with wickets in hand for bowlers defending a total.

Separate bat-facing vs bowl-defending emphasis is allowed *inside* scoring, but \(P\) itself remains match-state difficulty.

### Pressure intensity usage

- Primary analysis window: e.g. top quartile / \(P \ge P_{75}\) (threshold tuned & reported).
- Balls also keep continuous weight \(w(P)\) so 18 needed off 6 ≫ 55 needed off 60.

### What we will *not* do

- Pure death-over definition as the whole model.
- Hardcode famous players.
- Train and evaluate WP on the same matches without blocking.

---

## E. Batting & Bowling Scoring Architecture

### Layer 2 — Contextual Performance

Build **situation cells** (as fine as sample allows):

`innings × phase × wicket_band × (RRR_band or par_band) × [role_band]`

Role band from `batting_order` (e.g. 1–3 opener/top, 4–6 middle, 7+ finisher). Shrink cell means with Empirical Bayes when thin.

#### Batter metrics (pressure-weighted)

| Metric | Intent |
|---|---|
| Runs Above Expected (RAE) | \(r - \mathbb{E}[r \mid cell]\) |
| SR vs expected | strike-rate uplift in cell |
| Boundary conversion | 4/6 rate vs cell |
| Dismissal avoidance | wicket rate vs cell (run-outs attributed carefully) |
| Player baseline delta | pressure cell residual − non-pressure residual (**clutch vs self**) |

#### Bowler metrics (pressure-weighted)

| Metric | Intent |
|---|---|
| Runs prevented | \(\mathbb{E}[r_{bowl}] - r_{bowl}\) (correct extras) |
| Wickets above expected | bowler-credited wkts vs cell |
| Dot / boundary prevention | vs cell |
| Economy improvement | pressure vs own baseline in similar phases |
| Clutch vs self | pressure residual − non-pressure residual |

### Layer 3 — Reliability Engine

1. **Eligibility (to be tested, not fixed blindly):** e.g. ≥100–150 pressure legal balls AND ≥8–10 distinct pressure matches; report sensitivity at 80/120/180.
2. **Empirical Bayes** shrinkage of rate stats toward cell/population priors (\(K\) chosen by split-half or moment matching).
3. **Match-level consistency:** fraction of pressure matches with positive net residual; or match-cluster SD.
4. **Bootstrap stability:** 500–1000 match-cluster resamples → rank distribution, top-10 frequency, **1-SD lower-bound score** for final ordering (Sughan-style, adapted).
5. **Composite rating (interpretable):**

$$\mathrm{Score} = a\cdot z(\mathrm{RAE/RP}) + b\cdot z(\Delta_{\text{vs self}}) + c\cdot z(\mathrm{Consistency})$$

then apply reliability/lower-bound. Weights chosen by **stability + face-validity checks**, not by producing celebrity names.

Separate composites for batters and bowlers.

---

## F. Validation Strategy

| Test | Purpose |
|---|---|
| Match-blocked WP OOS (if WP used) | Prevent circular leverage |
| Pressure sanity checks | Extreme chase states score high; ball 1 of easy chase scores low |
| Paired same-player pressure vs non-pressure | Confirm pressure regime is harder on average |
| Bootstrap rank stability | Fragile names get low confidence labels |
| Sensitivity grid | Vary \(P\) threshold, \(K\), eligibility, weights, IPL↔T20I weighting |
| Competition split | IPL-only vs T20I-only leaderboards as diagnostics (not necessarily separate submissions) |
| Credit identity tests | Extras/wicket attribution unit tests |
| Temporal split | 2022–24 fit vs 2025–26 persistence (expect weak; report honestly) |

**Confidence labels** for report: Lock / Strong / Provisional based on bootstrap top-10 frequency.

If critical gates fail (à la Chauhan), we **say so** and still submit the most stable lower-bound ranking with caveats — honesty is a feature.

---

## G. Implementation Plan

### Modules

```
super_rr/
  01_audit.py              # schema, DQ report
  02_clean_state.py        # legal balls, extras, match state before ball
  03_pressure.py           # P1/P2 continuous pressure + intensity weights
  04_expectations.py       # contextual cell expectations + EB
  05_batter_ratings.py     # RAE, deltas, consistency
  06_bowler_ratings.py     # runs prevented, wkts, consistency
  07_reliability.py        # eligibility, EB, bootstrap, lower bounds
  08_validate.py           # sensitivity, stability, diagnostics
  09_export.py             # full tables + top10 + Unstop IDs
  report/                  # methodology doc + charts
```

### Dependencies

`pandas`, `numpy`, `scipy` (core). Optional later: `scikit-learn` (WP), `matplotlib`/`plotly` (figures). No dashboard required for Round 1 scoring.

### Expected outputs

1. Data quality report (counts, exclusions)
2. Delivery-level file with state + \(P\)
3. Full eligible batter/bowler rankings with components + CI
4. Final Top 10 each + player_id for Unstop
5. Bootstrap stability table
6. Sensitivity appendix
7. Methodology report with equations

### Execution order (after approval)

1. Implement audit + state reconstruction + unit tests for extras/wickets  
2. Pressure engine + distribution diagnostics  
3. Expectation cells + player residuals  
4. Reliability + bootstrap ranking  
5. Validation / sensitivity  
6. Freeze rankings + write report  

---

## H. Approved Final Decisions (locked)

| # | Decision | Locked choice |
|---|---|---|
| 1 | Win-probability leverage | **Conditional yes** — additive signal only after match-blocked OOS validation; foundation remains resource/time/rate pressure |
| 2 | Primary ranking criterion | **1-SD bootstrap lower bound** \(R_{\text{stable}}=\bar R-\sigma_{\text{bootstrap}}\); also report mean, SD, CIs; test 90% LCB alternative |
| 3 | Eligibility | Start **120 pressure balls + 8 matches**; sensitivity at 80/5, 150/10, 200/12 |
| 4 | IPL / T20I | **Normalize within competition**, then combine; test equal vs sample-proportional weights; single-competition players remain eligible with uncertainty penalty |
| 5 | Match scope | Exclude abandoned & no-result from final rankings; retain reconstructable revised-target matches; **exclude super overs** from primary model |

### Additional locked requirements

- Batting and bowling expectations must be **role- and context-adjusted**.
- Unit-test legal deliveries, extras attribution, wicket credit, and player-ID joins.
- No hardcoded / reputation-targeted rankings.
- Every shortlisted player must have an **explanation card**: pressure sample, expected vs actual, consistency, ranking uncertainty.
- Reproducibility: seed all resampling; document assumptions.

### Competition combination (revised)

1. Compute residuals / ratings separately against **IPL** and **T20I** (non-IPL T20 grouped with competition baseline rules documented in code).
2. Z-normalize (or pool-normalize) within each competition family.
3. Combine with weight scheme \(w\):
   - Equal: \(w_{\text{IPL}}=w_{\text{T20I}}=0.5\) when both present; else use available family only.
   - Sample-proportional: weights ∝ pressure balls in each family.
4. Apply EB / bootstrap on the combined score; single-family players automatically carry higher bootstrap variance → lower \(R_{\text{stable}}\).

### Pressure construction (revised)

$$P_{\text{base}} = f(\text{RRR/CRR gap, wickets, balls remaining, innings/par context})$$

$$P = \mathrm{clip}\big((1-\lambda) P_{\text{base}} + \lambda P_{\text{lev}}\big)$$

where \(\lambda=0\) unless OOS WP leverage passes validation gates; if it fails, ship \(P=P_{\text{base}}\) and document.
