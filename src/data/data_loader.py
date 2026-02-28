from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
import yaml

base_dir = Path(__file__).resolve().parent.parent.parent
config_dir = base_dir / "src" / "config"

train_path = base_dir / "data" / "train.csv"
test_path = base_dir / "data" / "test.csv"

preprocessing_path = config_dir / "preprocessing_config.yaml"
model_config_path = config_dir / "model_config.yaml"
best_hyperparams_path = config_dir / "best_hyperparams.yaml"

target = "SalePrice"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace({" ": "_"})
    return df

def data_split(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


