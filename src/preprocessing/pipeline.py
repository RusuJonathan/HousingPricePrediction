from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.preprocessing.transformers import FeatureEngineering, LotFrontageFiller, FillKnownNan
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.data.data_loader import load_yaml, preprocessing_path

config = load_yaml(preprocessing_path)

from sklearn import set_config
set_config(transform_output="pandas")

def build_encoding_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "ordinal_uncoded",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        categories=list(config["ORDINAL_COLS_RANK"].values()),
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ))
                ]),
                list(config["ORDINAL_COLS_RANK"].keys())
            ),
            (
                "ordinal_encoded",
                SimpleImputer(strategy="most_frequent"),
                config["ORDINAL_COLS_ENCODED"]
            ),
            (
                "nominal",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ]),
                config["NOMINAL_COLS"]
            ),

            (
                "numerical",
                SimpleImputer(strategy="median"),
                config["NUMERICAL_COLS"]
            )
        ],
        verbose_feature_names_out=False
    )

def build_data_preparation_pipeline():
    return Pipeline(steps=[
        ("lot_frontage_filler", LotFrontageFiller()),
        ("known_nan_filler", FillKnownNan()),
        ("imputer_encoder", build_encoding_preprocessor()),
        ("feature_engineering", FeatureEngineering()),
        ("scaler", StandardScaler()),
    ])
