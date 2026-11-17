# SupeRR Selector Season 2 — Pressure Reliability

Identify the most reliable batters and bowlers under pressure using the official IPL & T20I data pack (2022–2026).

## Live dashboard

**https://indrani2005.github.io/Analysis/**

- Top 10 batters & bowlers with explanation cards  
- Extended leaderboards  
- [Methodology report](https://indrani2005.github.io/Analysis/methodology.html)

## What this project claims

This does **not** invent Empirical Bayes, bootstrap ranking, or win-probability pressure.

**Contribution:** integrating those established tools with:

1. Role-aware situational expectations  
2. First-innings pressure modelling  
3. Within-competition normalisation (IPL vs T20I)  
4. Reliability-adjusted rankings (`R_stable = boot_mean − boot_sd`)

## Unstop submission (Top 10)

### Batters
1. Dinesh Karthik · 2. Tristan Stubbs · 3. Andre Russell · 4. Rahul Tewatia · 5. Tim David  
6. Marcus Stoinis · 7. Suryakumar Yadav · 8. Rinku Singh · 9. Namboori Tilak Varma · 10. David Miller

### Bowlers
1. Kagiso Rabada · 2. Josh Hazlewood · 3. Jasprit Bumrah · 4. Sunil Narine · 5. Rashid Khan (`4722`)  
6. Nathan Ellis · 7. Matheesha Pathirana · 8. Jofra Archer · 9. Kuldeep Yadav · 10. Varun Chakravarthy

## Repo layout

```
docs/                 # GitHub Pages dashboard
super_rr/             # Python modelling pipeline
tests/                # Attribution / pressure unit tests
outputs/              # Submission rankings + methodology
METHODOLOGY_PROPOSAL.md
requirements.txt
```

## Reproduce locally

Place the official Data Pack CSVs under `6af458274c1e8_data_pack/Data Pack/` (gitignored), then:

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m super_rr.run_pipeline
python -m super_rr.fix_and_rerank
```

## License / data

Competition data pack is not redistributed in this repository. Rankings are derived from the official Round 1 files only.
