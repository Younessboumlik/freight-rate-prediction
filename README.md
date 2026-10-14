# Freight Rate Prediction

Predict the posted rate (USD) of a freight load from its route, distance, equipment, weight and pickup date. The assessment instructions are in `freight-rate-ml-assessment.pdf`.

The whole analysis, model and deliverables are produced by one notebook, `Project.ipynb`.

## Results

| Model | MAE, time-ordered CV (USD) | MAE, held-out Sep-Oct test (USD) |
|---|---|---|
| Baseline: rate per mile by equipment | 206.7 | 229.1 |
| CatBoost, l1 loss, rate-per-mile target, tuned (final) | 107.9 | 164.6 |

The test error is higher than the CV error because the price level in Sep-Oct moved relative to `market_index`. The test figure is the honest estimate. Details are in the notebook summary.

## Repository

| File | Content |
|---|---|
| `Project.ipynb` | exploration, validation, modelling, final predictions |
| `requirements.txt` | Python dependencies |
| `score.py` | provided scorer, not modified |
| `train-test.csv`, `validation.csv`, `validation-predictions-template.csv`, `december-chart-inputs.csv` | provided data, not modified |
| `validation_predictions.csv` | final predictions for the 12,000 validation loads |
| `december_predictions.csv` | predictions for the fixed December route |
| `scorer_results/candidate_december.png` | December chart produced by `score.py` |
| `freight_rate_model.joblib` | final fitted pipeline |

## Run

Tested with Python 3.11.

```bash
python -m pip install -r requirements.txt
jupyter lab Project.ipynb
```

Run all cells from top to bottom. The provided data files and `score.py` must be in the same folder as the notebook. Seeds are fixed, so a second run gives the same results.

The hyperparameter search takes about 30 minutes, the rest a few minutes. To run without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Project.ipynb
```

The last cells call the scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

## Method

- **Validation follows time.** Sep-Oct is held out and used once. On Jan-Aug, cross-validation uses two-month windows moving forward, each trained on the six months before it. Anything learned from data (medians, fitted models) is fitted on the training fold only.
- **Metric.** MAE. About 1.4% of the prices are random outliers that cannot be predicted from the available columns, and they dominate RMSE.
- **Features.** Coordinates, distance, weight, equipment and `market_index`. Negative weights are made positive and missing values are filled with the training median (or the median of the same day for `market_index`). `quote_signal` is removed because its relation with the price changes sign from month to month.
- **Model.** CatBoost with an l1 loss, trained on the rate per mile and converted back to a price. Hyperparameters come from a two-stage random search on the time-ordered folds. The final model is retrained on all labelled data (Jan-Oct) before predicting `validation.csv`.

## December chart

`december-chart-inputs.csv` has no coordinates and no `market_index`. The coordinates are those of the route in the labelled data. For each day, `market_index` is the median of that day in `validation.csv`. The prediction varies by less than 2% over the month, and the chart axis is zoomed.

## Limits

The price level moved in Sep-Oct and may move again in Nov-Dec, which cannot be checked without labels. With one year of data, no seasonal pattern can be learned, so no calendar feature is used.
