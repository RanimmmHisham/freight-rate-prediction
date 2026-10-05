# Freight rate prediction

Predict `posted_rate` for 12,000 loads in Nov-Dec 2025 from 48,000 labelled loads dated Jan-Oct 2025, then predict a fixed Lexington to Fort Wayne Dry Van load for every day of December.

Full write-up with figures: [`reports/Freight_Rate_Report.docx`](reports/Freight_Rate_Report.pdf).

## Approach in one paragraph

Validation is by time, not at random, because the target period is entirely after the training data. The model predicts log rate per mile in two stages. Stage 1 is a small gradient-boosted tree on static load attributes (log distance, |weight|, equipment, pickup/delivery coordinates) trained after setting aside the 1.4% of labels that look corrupt. Stage 2 is a one-parameter linear correction for how far that day's `market_index` is from its month's average. City names are not used, so cities never seen in training are handled the same way as known ones.

## Results (mean of 4 forward-in-time folds, Jul-Oct)

| Model | MAE $ | Median APE | Clean MAPE |
|---|---|---|---|
| Median rate/mile by equipment x distance band | 134.8 | 3.20% | 3.71% |
| Robust linear | 118.9 | 2.58% | 2.94% |
| GBM, static features | 104.2 | 1.86% | 2.11% |
| **Final: GBM + month-deviation market adjustment** | **101.0** | **1.75%** | **1.99%** |

"Clean" excludes the ~1.4% corrupt-looking labels, which no model can predict; all other columns include them. The final scoring metric is computed by Spotter after submission, so these are my own cross-validation numbers, not a leaderboard score.

Things that did **not** work, kept in the repo as evidence: using `market_index` as an ordinary model input (worse out of time: 2.11% -> 3.17% clean MAPE), city identity features, an absolute-error loss, `quote_signal`, day of week, hyperparameter tuning (flat across 12 settings).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
# put the provided CSVs in data/ as train_test.csv, validation.csv,
# validation_predictions_template.csv, december_chart_inputs.csv

python src/data_checks.py       # data-quality numbers quoted in the report (seconds)
python src/experiments.py       # all experiments, writes results/*.csv (~10 min, 2 cores)
python src/train_predict.py     # final fit -> validation_predictions.csv + December predictions
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs_predicted.csv
python src/make_figures.py      # report figures
python tests/test_sanity.py     # split / leakage / output-format checks
```

Everything is seeded; reruns reproduce the same numbers. Tested on Python 3.13, scikit-learn 1.9.

## Layout

```
src/features.py       cleaning + features (label-free, shared by every script)
src/models.py         baselines, GBM, two-stage MarketAdjusted model, metrics
src/harness.py        data loading, temporal folds, scoring helpers
src/experiments.py    every experiment behind a modelling decision
src/train_predict.py  final fit and output files
src/data_checks.py    EDA numbers quoted in the report
tests/                split and leakage guards
results/              CSV + text output of the runs (the evidence for the report)
reports/              report (docx/pdf), figures, script that builds it
```

## Things to know

- **December chart input.** `december_chart_inputs.csv` has no `market_index`, so each date takes its value from `validation.csv` (a feature column, no labels). Without that the curve would be nearly flat at about $821. The weekly cycle in the chart comes from the index's own day-of-week pattern; the model does not know about Christmas.
- **Biggest risk.** Each month has its own rate level that `market_index` does not explain (standard deviation about 3 points). With one year of data I cannot tell whether Nov-Dec carries a seasonal offset.
- **Model selection** used the same four folds that are reported. The large decisions are not close; the small ones (for example 1.95% vs 1.99%) should not be over-read.
- **Library versions.** Predictions move by about 0.1% between scikit-learn versions (checked on 1.9.1 and 1.6.1; the validation mean is identical at $2,357). The numbers in the report come from 1.9.1.
- **Windows.** A joblib warning about "physical cores" is harmless; set `LOKY_MAX_CPU_COUNT` to silence it.
- `data/` is git-ignored; it holds the assessment files.
