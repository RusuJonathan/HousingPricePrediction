import pandas as pd
from typing import Dict, List
from src.data.data_loader import load_yaml, preprocessing_path
import numpy as np

preprocessing_config = load_yaml(preprocessing_path)

KNOWN_NAN_COLS = preprocessing_config["KNOWN_NAN_COLS"]
QUALITY_COLS = preprocessing_config["QUALITY_COLS"]

def fill_known_nan(df: pd.DataFrame,
                   known_nans: Dict[str, str] = KNOWN_NAN_COLS) -> pd.DataFrame:
    """
    Fill known NaN values in categorical columns with appropriate placeholders.

    For example, if 'Alley' is NaN, it means there is no alley, so we replace NaN with 'NoAlley'.

    :param df: pandas DataFrame
    :param known_nans: dict mapping column names to filler strings
    :return: copy of df with NaNs filled
    """
    return df.fillna(value=known_nans)

def fill_mas_vnr_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills Mas Vnr Type, if the area of Mas Vnr Area is 0 then that's means there's no Masonery, so we can
    fill with 0
    :return: DataFrame with filled MasVnrTyep
    """
    df = df.copy()
    mas_vnr_area_is_0 = df["MasVnrArea"] == 0
    df.loc[mas_vnr_area_is_0, "MasVnrType"] = "NoMas"
    return df

def aggregate_features(df: pd.DataFrame,
                       features_to_sum: Dict[str, list]) -> pd.DataFrame:
    """
    Create new features by summing existing features
    :param df: pandas DataFrame
    :param features_to_sum:
    :return: DataFrame with new features
    """
    df = df.copy()
    for new_feature, feature_list in features_to_sum.items():
        df[new_feature] = df[feature_list].sum(axis=1)

    return df

def add_total_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature called TotalBathRooms, we count the total nb of bathrooms in the house,
    where we put a 0.5 weight on half baths, and 1 on full baths

    :param df: original pandas DataFrame
    :return: pandas DataFrame
    """
    df = df.copy()
    df["TotalBathrooms"] = df[["FullBath", "BsmtFullBath"]].sum(axis=1) + 0.5 * df[["HalfBath", "BsmtHalfBath"]].sum(axis=1)
    return df

def add_bed_per_bath(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature called BedPerBath, this ratio represents the numbers per baths
    :param df: Original pandas Dataframe
    :return: pandas Dataframe with new feature
    """
    df = df.copy()
    assert "TotalBathrooms" in df.columns, "TotalBathrooms must exist before BedPerBath"

    df["BedPerBath"] = df["TotRmsAbvGrd"] / df["TotalBathrooms"].replace(0, np.nan)
    median_bed_per_bath = df["BedPerBath"].median()
    df["BedPerBath"] = df["BedPerBath"].fillna(median_bed_per_bath)
    return df

def log_transform_features(df: pd.DataFrame,
                           features_to_log_transform: List[str]) -> pd.DataFrame:
    """
    Log transforms the features, to remove the skewness from them
    :param df: original pandas dataframe
    :param features_to_log_transform:
    :return: pandas DataFrame with logged transform features
    """

    df = df.copy()
    for feature in features_to_log_transform:
        if feature not in df.columns:
            raise ValueError(f"Column {feature} not in dataframe")
        df[feature] = np.log1p(df[feature])

    return df

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds age features linked to the house
    :param df:
    :return:
    """
    df = df.copy()
    df["AgeOfHouse"] = df["YrSold"] - df["YearBuilt"]
    df["TimeSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    return df

def add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary features, i.e for pool, we create a new column with a 1 if there's a pool, 0 if not
    :param df: original pandas Dataframe
    :return:  DataFrame with new features
    """
    df = df.copy()
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFirePlace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasGarage"] = (df["GarageCars"] > 0).astype(int)
    df["Remodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ratio features, for example, ratio surface land, the ratio between the land and the livable space
    :param df:
    :return:
    """
    df = df.copy()
    df["RatioSurfLand"] = df["GrLivArea"] / df["LotArea"]
    df["SurfacePerCar"] = df["GarageArea"] / df["GarageCars"].replace(0, np.nan)
    df["SurfacePerCar"] = df["SurfacePerCar"].fillna(0)
    df["SurfacePerRoom"] = df["TotalSF"] / df["BedroomAbvGr"].replace(0, np.nan)
    df["SurfacePerRoom"] = df["SurfacePerRoom"].fillna(0)
    return df

def quality_score_feature(df: pd.DataFrame,
                          qual_features: List[str] = QUALITY_COLS) -> pd.DataFrame:
    """
    Score of quality accross all elements of the house
    :param df:
    :param qual_features:
    :return:
    """
    df = df.copy()
    df["TotalQualityScore"] = df[qual_features].mean(axis=1)
    return df
