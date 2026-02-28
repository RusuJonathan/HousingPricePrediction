import optuna
import pandas as pd
from typing import Dict, Callable, Any
from src.models.models_builders import set_base_model_params, set_stacking_model_params
from sklearn.model_selection import cross_val_score
from pathlib import Path
from optuna.study import Study
import yaml


from src.models.models_builders import build_pipeline
from src.data.data_loader import load_yaml, model_config_path, best_hyperparams_path, train_path, load_data, target
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

def build_objective_function(model: Any,
                             x_train: pd.DataFrame,
                             y_train: pd.Series,
                             model_config: Dict[str, Dict],
                             base_model: bool = True,
                             cv: int = 5) -> Callable:
    def objective(trial):

        trial_params = {}

        for param_name, param_info in model_config.items():
            distribution = param_info["distribution"]
            if distribution == "IntUniformDistribution":
                trial_params[param_name] = trial.suggest_int(param_name, low=param_info["low"], high=param_info["high"])
            elif distribution == "LogUniformDistribution":
                trial_params[param_name] = trial.suggest_float(param_name, low=param_info["low"], high=param_info["high"], log=True)
            elif distribution == "UniformDistribution":
                trial_params[param_name] = trial.suggest_float(param_name, low=param_info["low"], high=param_info["high"])
            else:
                trial_params[param_name] = trial.suggest_categorical(param_name, param_info["categories"])

        if base_model:
            set_base_model_params(pipeline=model, param=trial_params)
        else:
            set_stacking_model_params(stacking=model, param=trial_params)

        scores = cross_val_score(
            estimator=model,
            X=x_train,
            y=y_train,
            scoring="neg_mean_absolute_error",
            cv=cv)

        return - scores.mean()
    return objective

def run_hyperparameter_optimization(
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_config: Dict[str, Dict],
        n_trials: int,
        base_model: bool = True,
        cv: int = 5):

    objective_func = build_objective_function(
        model=model,
        x_train=X,
        y_train=y,
        model_config=model_config,
        base_model=base_model,
        cv=cv)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=n_trials)
    return study

def save_best_hyperparameters(
        study : Study,
        model_name: str,
        filepath: Path = best_hyperparams_path) -> None:

    model_param = {
        model_name : study.best_params
    }

    try:
        with open(filepath, "r") as f:
            existing = yaml.safe_load(f) or {}
    except FileNotFoundError:
        existing = {}

    existing.update(model_param)

    with open(filepath, "w") as f:
        yaml.safe_dump(existing, f)


if __name__ == "__main__":
    pipeline = build_pipeline(SVR)

    config = load_yaml(model_config_path)
    df = load_data(train_path)

    x_train = df.drop(columns=[target])
    y_train = df[target]

    study = run_hyperparameter_optimization(model=pipeline,
                                    X=x_train,
                                    y=y_train,
                                    model_config=config["svr"],
                                    n_trials=300)

