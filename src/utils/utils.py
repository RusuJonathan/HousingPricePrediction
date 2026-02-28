import pandas as pd
from src.data.data_loader import target
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corrmat(df: pd.DataFrame) -> None:
    numerical_cols = df.select_dtypes(["int", "float"]).columns
    corrmat = df[numerical_cols].corr()
    plt.figure(figsize=(16,16))
    sns.heatmap(data=corrmat,
                fmt="0.2f",
                annot=True,
                square=True,
                linewidth=1,
                cbar=False,
                cmap="viridis")
    plt.tight_layout()
    plt.show()


def plot_dist(df: pd.DataFrame) -> None:
    numerical_cols = df.select_dtypes(["int", "float"]).columns
    fig, axes = plt.subplots(nrows=len(numerical_cols), figsize=(10, len(numerical_cols) * 6))
    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        sns.kdeplot(
            data=df,
            x=col,
            ax=ax
        )
        ax.set(xlabel="", ylabel="")
        ax.set_title(f"\n{col}")

    plt.tight_layout()
    plt.show()

def cat_boxplot(df: pd.DataFrame) -> None:
    categorical_cols = df.select_dtypes(["object"]).columns
    fig, axes = plt.subplots(nrows=len(categorical_cols), figsize=(10, len(categorical_cols) * 6))
    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x=col,
            y=target,
            ax=ax
        )
        if df[col].nunique() > 5:
            ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()

def missing_summary(df, label='Train'):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({'Missing': miss, 'Missing (%)': miss_pct}).rename_axis(f'{label} — Column')
