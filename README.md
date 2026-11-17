# SupeRR Selector Season 2 — Pressure Reliability

Data-driven system to measure how batters and bowlers perform in high-pressure match states versus contextual expectation, then rank by reliability across resampled matches.

Built for Rajasthan Royals SupeRR Selector Season 2 (Round 1) using the official IPL & T20I ball-by-ball data pack (2022–2026).

## Live dashboard

**https://indrani2005.github.io/Analysis/**

- Interactive pressure rankings and player explanation cards  
- Extended leaderboards  
- [Methodology report](https://indrani2005.github.io/Analysis/methodology.html)

## Contribution (what this is / isn’t)

This project does **not** claim to invent Empirical Bayes shrinkage, bootstrap ranking, or win-probability / leverage pressure. Those are established sports-analytics tools.

**The contribution is their integration for this problem:**

1. **Role-aware situational expectations** — finishers are not judged like openers  
2. **First-innings pressure modelling** — not chase-only / death-over heuristics  
3. **Within-competition normalisation** — IPL and T20I scored separately, then combined  
4. **Reliability-adjusted ranking** — primary sort by bootstrap lower bound  
   \(R_{\text{stable}} = \bar{R} - \sigma\)

Central question: when the contest is genuinely difficult, does the player beat what a comparable player would be expected to produce in that situation — and can they repeat it?

## Method sketch

| Layer | What it does |
|---|---|
| Pressure engine | Continuous pre-ball difficulty (rate / resources / time / par) + conditional live-contest leverage |
| Performance | Actual vs expected residuals + absolute pressure execution |
| Fairness | Legal-ball / extras / wicket attribution; role filters; IPL–T20I handling |
| Reliability | Empirical Bayes shrinkage, eligibility thresholds, match-cluster bootstrap |

Full equations, validation, and limitations are in the methodology page / `outputs/METHODOLOGY_REPORT.html`.

## Repo layout

```
docs/                      GitHub Pages dashboard + methodology
super_rr/                  Modelling pipeline
tests/                     Attribution & pressure unit tests
outputs/                   Generated tables + report artifacts
METHODOLOGY_PROPOSAL.md    Locked modelling decisions
requirements.txt
```

Final ranked outputs are produced by the pipeline (see dashboard). They are intentionally not pasted as a plain name list in this README.

## Reproduce locally

1. Place the official Data Pack CSVs under `6af458274c1e8_data_pack/Data Pack/` (gitignored — not redistributed here).  
2. Run:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m super_rr.run_pipeline
python -m super_rr.fix_and_rerank
```

## Data note

The competition CSV pack is **not** included in this repository. All reported rankings must be regenerated from the official Round 1 files.
