# 🏠 House Prices — Advanced Regression Techniques

[![Python](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Stacking ensemble with Optuna hyperparameter optimization achieving a **top ~4% ranking** on the Kaggle leaderboard.

---

## 📊 Results

| Metric | Score |
|---|---|
| Kaggle Public Score (RMSLE) | **0.12124** |
| Leaderboard Score | **0.11981** |
| Leaderboard Rank | **~Top 4%** (≈159 / 4010) |

---

## 📁 Project Structure

```
house-prices/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── config/
│   │   ├── preprocessing_config.yaml   # Column types, ordinal ranks, features to engineer
│   │   ├── model_config.yaml           # HPO search spaces per model
│   │   └── best_hyperparams.yaml       # Saved best hyperparameters (auto-generated)
│   ├── data/
│   │   └── data_loader.py              # CSV loading, train/test split, YAML config loader
│   ├── preprocessing/
│   │   ├── pipeline.py                 # Scikit-learn preprocessing pipeline builder
│   │   ├── transformers.py             # Custom sklearn transformers
│   │   └── preprocessing_utils.py      # Feature engineering helper functions
│   └── models/
│       ├── model.py                    # Main Model class (fit / predict / optimize)
│       ├── models_builders.py          # Pipeline & stacking model factory functions
│       └── hpo_tuner.py                # Optuna-based HPO objective & study runner
├── notebooks/
│   └── eda_house_prices.ipynb          # Exploratory Data Analysis
├── main.py                             # Entry point: train → optimize → predict → submit
└── README.md
```

---

## 🧠 Approach

### Preprocessing pipeline

Each model is wrapped in a unified `sklearn.Pipeline` that applies the following steps in order:

1. **`LotFrontageFiller`** — Imputes `LotFrontage` using the median grouped by `MSZoning` × `LotArea` quintile bin. The grouping is learned at fit time and applied at transform time, preventing data leakage.

2. **`FillKnownNan`** — Replaces structurally meaningful NaN values (e.g. `PoolQC=NaN` → `"NoPool"`) and handles `MasVnrType` based on `MasVnrArea`.

3. **`ColumnTransformer`** — Handles column types in parallel:
   - *Ordinal (unencoded)*: imputed → `OrdinalEncoder` with explicit rank categories
   - *Ordinal (pre-encoded)*: median imputation
   - *Nominal*: imputed → `OneHotEncoder`
   - *Numerical*: median imputation

4. **`FeatureEngineering`** — Creates new features:
   - `TotalSF`, `TotalPorchSF` (area aggregations)
   - `TotalBathrooms` (full + 0.5 × half baths)
   - `AgeOfHouse`, `TimeSinceRemodel`, `GarageAge`
   - `HasPool`, `HasFirePlace`, `HasGarage`, `Remodeled` (binary flags)
   - `RatioSurfLand`, `SurfacePerCar`, `SurfacePerRoom`, `BedPerBath` (ratios)
   - `TotalQualityScore` (mean across quality columns)
   - Log-transforms on highly skewed numerical features

5. **`StandardScaler`** — Normalises all features.

6. **`TransformedTargetRegressor`** — Applies np.log1p to the target at training time and np.expm1 at prediction time, normalizing the skewed SalePrice distribution and aligning the regression objective with the RMSLE competition metric.
### Modeling — Stacking Ensemble

A `StackingRegressor` combines six base learners, each wrapped in its own preprocessing pipeline:

| Model | Library |
|---|---|
| ElasticNet | scikit-learn |
| SVR | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| CatBoost | catboost |
| Random Forest | scikit-learn |

**Meta-learner:** `ElasticNet`

### Hyperparameter Optimization

All hyperparameters are tuned with **Optuna** using:
- 300 trials per base model
- 5-fold cross-validation
- `neg_mean_absolute_error` as the objective
- Search spaces defined in `model_config.yaml` (supports `IntUniform`, `LogUniform`, `Uniform`, and `Categorical` distributions)

---

---

## 📦 Dependencies

Dependencies are managed with **Poetry** (`pyproject.toml`). Main packages:

```
scikit-learn
xgboost
lightgbm
catboost
optuna
pandas
numpy
pyyaml
matplotlib
seaborn
scipy
```

---

## 📓 EDA Highlights

Key findings from `notebooks/eda_house_prices.ipynb`:

- `SalePrice` is right-skewed → log transformation is useful
- Many NaN values are meaningful, not random missing data
- `OverallQual`, `GrLivArea`, and `GarageCars` are the top 3 predictors
- Neighborhood creates strong price tiers — up to 3× price difference between areas
- Remodeled houses command a significant median price premium
- Age-related features (`AgeOfHouse`, `TimeSinceRemodel`) provide useful signal beyond raw year columns

---

## 📄 License

This project is licensed under the MIT License.
