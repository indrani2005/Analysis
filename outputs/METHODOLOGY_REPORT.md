# SupeRR Selector Season 2 — Methodology Report

**Identifying the Most Reliable Batters and Bowlers Under Pressure**

Round 1 · Official IPL & T20I data pack (2022–2026) · Seed 42

---

## Executive summary

This report ranks players by **pressure-adjusted reliability**: whether they repeatedly outperform what a role-matched player would be expected to produce in the same match state.

We do **not** claim to invent Empirical Bayes shrinkage, bootstrap ranking, or win-probability / leverage pressure. Those are established tools in sports analytics. **Our contribution is how they are integrated** for this problem:

1. **Role-aware situational expectations** (top / middle / lower order; pace / spin)  
2. **First-innings pressure modelling** (par and resources, not chase-only)  
3. **Within-competition normalisation** (IPL vs T20I) before combining evidence  
4. **Reliability-adjusted rankings** via a bootstrap lower bound, with live-contest pressure and a par gate  

Central question:

> When conditions are genuinely difficult, does this player beat contextual expectation — and can they do it again?

---

## 1. What we are measuring

Raw strike rate and economy treat every ball equally. A quiet powerplay single and a required-rate scramble in a live chase are not the same skill test.

We therefore:

- score **pre-delivery** match difficulty continuously  
- compare **actual vs expected** outcomes in similar situations  
- rank by a **stable lower bound**, not a single peak performance  

Reputation is never used as an input.

---

## 2. Data scope and credit rules

| Step | Decision | Outcome |
|---|---|---|
| Source | Official Data Pack CSVs | 200,999 deliveries · 883 matches |
| Ball coverage | Matches with ball-by-ball rows | 867 matches |
| Exclusions | Abandoned / No Result | Removed from final rankings |
| Super overs | Innings ≥ 3 | Excluded from primary model |
| Revised targets | Kept if state/target reconstructable | Retained |
| Scoped analysis set | Innings 1–2 | **199,208** deliveries · **843** matches |

**Attribution (unit-tested)**

- Legal delivery: not wide, not Overstep (no-ball)  
- Bat runs: off-the-bat `runs`  
- Bowler runs: bat runs + wide/no-ball extras; **byes and leg-byes excluded**  
- Bowler wickets: Caught, Bowled, LBW, Stump, Hit Wicket only  
- Identity: always `player_id` (display-name collisions exist, including two “Rashid Khan” IDs)

---

## 3. Pressure engine

Pressure uses **information available before the ball only** (no look-ahead).

### 3.1 Transparent base score \(P_{\text{base}}\)

**Second innings (chase)** — combination of:

- required vs current rate gap  
- wickets in hand / balls remaining (resource stress)  
- deadline stress as balls run out  
- closeness when the equation is tight  

**First innings (setting)** — combination of:

- projected finish vs competition-family **par**  
- wicket-collapse stress  
- late-innings intensity (scoring obligation or defending a total)  

First-innings modelling is intentional: pressure is not defined as “death overs in a chase” alone.

### 3.2 Live-contest leverage (conditional)

A match-blocked logistic win-probability model supplies a **leverage** signal. Out-of-fold checks: AUC ≈ 0.80, Brier ≈ 0.18 → leverage is enabled as an **additive** component, not the sole definition of pressure.

Final pressure also **down-weights dead contests** (e.g. hopeless late chases with extreme required rates), so “pressure” means a live contest, not a decided game.

High-pressure balls are the upper tail of the continuous score (top ~20% in the final freeze), with continuous intensity weights above the cut.

---

## 4. Performance vs expectation

### 4.1 Role-aware context cells

Expected outcomes are cell means, shrunk with standard Empirical Bayes toward competition-family priors:

| Role | Cell factors |
|---|---|
| Batter | Family × innings × phase × wicket band × **role band** × RRR/par band |
| Bowler | Family × innings × phase × wicket band × pace/spin × RRR/par band |

Role bands follow batting order (top 1–3, middle 4–6, lower 7+). Finishers are not judged against openers.

**Delivery residual (positive = good for that role)**

- Batter: runs above expected, minus a run-valued dismissal penalty  
- Bowler: runs prevented vs expected, plus credit for bowler-attributed wickets  

### 4.2 Absolute execution pillar

Alongside residuals we use pressure strike rate (batters) and pressure economy / wicket rate (bowlers), z-scored **within competition** among players with adequate samples (micro-samples excluded from the z pool).

### 4.3 Competition normalisation

IPL and T20I are **not** blindly pooled at the raw level:

1. Compute metrics inside each family  
2. Normalise within family  
3. Combine with sample-proportional weights when both exist  
4. Single-competition players remain eligible; uncertainty is handled by the reliability layer  

Shortlist freeze additionally requires a meaningful **IPL pressure sample** so the Unstop lists are grounded in the competition most relevant to franchise selection, without deleting T20I evidence from the model.

### 4.4 Role eligibility

- Batters: sufficient balls faced, batting share of workload, median order not a pure No.11  
- Bowlers: sufficient balls bowled, bowling share of workload  

---

## 5. Reliability-adjusted ranking

Composite score (final freeze weights):

\[
R = 0.35\,z(\text{RAE}) + 0.35\,z(\text{Execution}) + 0.05\,z(\text{Clutch}_{\text{capped}}) + 0.25\,f(\text{Consistency})
\]

Clutch (pressure residual − non-pressure residual) is **lightly weighted and capped** so consistently excellent players are not penalised for also being good in ordinary conditions.

Empirical Bayes shrinks noisy rates toward a neutral prior by sample size. A **par gate** requires positive contextual residual under pressure.

**Primary ranking metric**

\[
R_{\text{stable}} = \bar{R}_{\text{bootstrap}} - \sigma_{\text{bootstrap}}
\]

Match-cluster bootstrap (matches resampled with replacement). We also record bootstrap mean, SD, and a 90% lower bound.

**Eligibility freeze:** ≥ 200 pressure balls and ≥ 12 distinct pressure matches (sensitivity also checked at 120/8 and 150/10).

---

## 6. Final Unstop rankings

### Batters

| Rank | Player | Player ID | Pressure balls | IPL pressure balls | \(R_{\text{stable}}\) |
|---:|---|---:|---:|---:|---:|
| 1 | Dinesh Karthik | 143 | 385 | 264 | 0.664 |
| 2 | Tristan Stubbs | 20702 | 271 | 198 | 0.561 |
| 3 | Andre Russell | 1318 | 327 | 271 | 0.542 |
| 4 | Rahul Tewatia | 3926 | 237 | 237 | 0.455 |
| 5 | Tim David | 6396 | 408 | 301 | 0.455 |
| 6 | Marcus Stoinis | 2639 | 341 | 256 | 0.444 |
| 7 | Suryakumar Yadav | 1699 | 338 | 216 | 0.352 |
| 8 | Rinku Singh | 4708 | 406 | 290 | 0.310 |
| 9 | Namboori Tilak Varma | 7897 | 475 | 332 | 0.304 |
| 10 | David Miller | 1077 | 418 | 357 | 0.276 |

### Bowlers

| Rank | Player | Player ID | Pressure balls | IPL pressure balls | \(R_{\text{stable}}\) |
|---:|---|---:|---:|---:|---:|
| 1 | Kagiso Rabada | 3628 | 321 | 266 | 0.715 |
| 2 | Josh Hazlewood | 830 | 326 | 213 | 0.617 |
| 3 | Jasprit Bumrah | 2933 | 455 | 329 | 0.613 |
| 4 | Sunil Narine | 781 | 231 | 231 | 0.602 |
| 5 | Rashid Khan | 4722 | 596 | 451 | 0.597 |
| 6 | Nathan Ellis | 7920 | 287 | 105 | 0.473 |
| 7 | Matheesha Pathirana | 8661 | 398 | 269 | 0.461 |
| 8 | Jofra Archer | 3298 | 243 | 139 | 0.404 |
| 9 | Kuldeep Yadav | 3644 | 423 | 320 | 0.371 |
| 10 | Varun Chakravarthy | 6256 | 264 | 189 | 0.367 |

Full player explanation cards (sample, expected vs actual, consistency, uncertainty) accompany this report.

---

## 7. Validation and limitations

**Validation**

- Unit tests for legal balls, extras, wicket credit, and player-ID collisions  
- Match-blocked WP hold-out metrics before enabling leverage  
- Pressure sanity checks (live tight chases ≫ easy starts / dead games)  
- Eligibility and competition-weight sensitivity  

**Limitations**

- T20 pressure skill is noisy; temporal persistence is expected to be weak  
- Venue and opposition adjustments are limited  
- First-innings par is competition-family based, not a full venue model  
- Rankings reflect this data pack and these modelling choices — not reputation  

---

## 8. Reproducibility

```text
pip install -r requirements.txt
python -m pytest tests -q
python -m super_rr.run_pipeline
python -m super_rr.fix_and_rerank
```

Assumptions and locked decisions are documented in `METHODOLOGY_PROPOSAL.md` and `super_rr/config.py`.
