import pandas as pd
from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin
from src.preprocessing.preprocessing_utils import (fill_known_nan,
                                                   fill_mas_vnr_type,
                                                   aggregate_features,
                                                   add_total_bathrooms,
                                                   add_bed_per_bath,
                                                   log_transform_features,
                                                   add_age_features,
                                                   add_binary_features,
                                                   add_ratio_features,
                                                   quality_score_feature)
from src.data.data_loader import load_yaml, preprocessing_path
config = load_yaml(preprocessing_path)

class FillKnownNan(BaseEstimator, TransformerMixin):
    def __init__(self, config: Dict = config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = fill_known_nan(X, self.config["KNOWN_NAN_COLS"])
        X = fill_mas_vnr_type(X)
        return X

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, config: Dict = config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = aggregate_features(X, self.config["FEATURES_TO_SUM"])
        X = add_total_bathrooms(X)
        X = add_bed_per_bath(X)
        X = log_transform_features(X, self.config["LOG_FEATURES"])
        X = add_age_features(X)
        X = add_binary_features(X)
        X = add_ratio_features(X)
        X = quality_score_feature(X, self.config["QUALITY_COLS"])
        return X

class LotFrontageFiller(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            mszoning: str = "MSZoning",
            lot_frontage: str = "LotFrontage",
            lot_area: str = "LotArea",
            q: int = 5
            ):
        self.mszoning = mszoning
        self.lot_frontage = lot_frontage
        self.lot_area = lot_area
        self.q = q

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        X["LotAreaBin"] = pd.qcut(X[self.lot_area],
                                  q=self.q,
                                  duplicates="drop")
        self.lot_area_bin = X["LotAreaBin"].cat.categories
        self.median_filler = X.groupby([self.mszoning, "LotAreaBin"])[self.lot_frontage].median()

        self.global_median = X[self.lot_frontage].median()
        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()
        X["LotAreaBin"] = pd.cut(X[self.lot_area], bins=self.lot_area_bin)

        X = X.merge(
            self.median_filler.rename("GroupMedian"),
            how="left",
            left_on=[self.mszoning, "LotAreaBin"],
            right_index=True
        )

        X[self.lot_frontage] = X[self.lot_frontage].fillna(X["GroupMedian"])
        X[self.lot_frontage] = X[self.lot_frontage].fillna(self.global_median)

        return X.drop(columns=["LotAreaBin", "GroupMedian"], errors="ignore")



