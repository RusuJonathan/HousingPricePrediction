from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pathlib import Path
from typing import Dict

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import StackingRegressor

from src.models.models_builders import build_pipeline, set_base_model_params, set_stacking_model_params, build_stacking_model
from src.models.hpo_tuner import run_hyperparameter_optimization, save_best_hyperparameters

from src.data.data_loader import load_yaml, load_data, train_path, model_config_path
from sklearn.base import BaseEstimator, RegressorMixin

class Model(BaseEstimator, RegressorMixin):
    def __init__(self, config: Dict, n_trials: int=300):
        self.save_best_hyperparameters = save_best_hyperparameters
        self.config = config
        self._build_pipeline = build_pipeline
        self._run_hyperparameter_optimization = run_hyperparameter_optimization
        self._build_stacking_model = build_stacking_model
        self.set_base_model_params = set_base_model_params
        self.set_stacking_model_params = set_stacking_model_params
        self._init_models()
        self.n_trials = n_trials
        self.stacking_model = StackingRegressor(
            estimators=self.estimators,
            final_estimator=ElasticNet(max_iter=10000)
        )

    def _init_models(self):
        self.elasticnet = ("elasticnet", self._build_pipeline(ElasticNet, {"max_iter" : 10000}))
        self.svr = ("svr", self._build_pipeline(SVR))
        self.xgboost = ("xgboost", self._build_pipeline(XGBRegressor, {"verbosity" : 0}))
        self.lgbm = ("lgbm", self._build_pipeline(LGBMRegressor, {"verbosity" : -1}))
        self.catboost = ("catboost", self._build_pipeline(CatBoostRegressor, {"verbose": 0}))
        self.randomforest = ("random_forest", self._build_pipeline(RandomForestRegressor))
        self.estimators = [
            self.elasticnet,
            self.svr,
            self.xgboost,
            self.lgbm,
            self.catboost,
            self.randomforest
        ]


    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series):

        for model_name, model in self.estimators:
            study = self._run_hyperparameter_optimization(
                model=model,
                X=X,
                y=y,
                model_config=self.config[model_name],
                n_trials=self.n_trials
            )
            self.set_base_model_params(model, study.best_params)
            #self.save_best_hyperparameters(study=study, model_name=model_name)


        #self.stacking_model = self._build_stacking_model(estimators=estimators, final_model_estimator=ElasticNet())
        # study = self._run_hyperparameter_optimization(
        #     model=self.stacking_model,
        #     X=X,
        #     y=y,
        #     model_config=self.config["elasticnet"],
        #     base_model=False,
        #     n_trials=self.n_trials
        # )
        #
        # self.set_stacking_model_params(self.stacking_model, study.best_params)

    def load_param(self, hyperparameter_config: Dict):
        for model_name, pipeline in self.stacking_model.estimators:
            pipeline.named_steps["model"].regressor.set_params(**hyperparameter_config[model_name])

        return self


    def get_all_params(self):
        all_params = {}

        for name, pipeline in self.stacking_model.estimators:
            model_step = pipeline.named_steps["model"]
            all_params[name] = model_step.get_params()

        meta = self.stacking_model.final_estimator

        all_params["meta_model"] = meta.get_params()

        return all_params


    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        predictions = self.stacking_model.predict(X)
        return predictions



if __name__== "__main__":
    config = load_yaml(model_config_path)
    df = load_data(train_path)
    x_train = df.drop(columns=["SalePrice"])
    y_train =df["SalePrice"]

    model = Model(config=config, n_trials=50)
    model.optimize_hyperparameters(x_train, y_train)

    scores = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="neg_mean_absolute_error",
        cv=5)
    print(-scores.mean())

