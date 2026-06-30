"""Shared plotting helpers for HR attrition EDA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def attrition_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute count and attrition rate for a categorical column."""
    summary = (
        df.groupby(column, observed=True)
        .agg(total=("Attrition_Flag", "count"), leavers=("Attrition_Flag", "sum"))
        .assign(rate=lambda x: (x["leavers"] / x["total"] * 100).round(1))
        .sort_values("rate", ascending=False)
    )
    return summary


def plot_attrition_rate(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str | None = None,
    figsize: tuple[int, int] = (10, 5),
    horizontal: bool = False,
    save_path=None,
) -> pd.DataFrame:
    """Bar chart of attrition rate by category; returns summary table."""
    summary = attrition_rate_table(df, column)
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        sns.barplot(
            y=summary.index.astype(str),
            x=summary["rate"],
            hue=summary.index.astype(str),
            palette="RdYlGn_r",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Attrition Rate (%)")
        ax.set_ylabel(xlabel or column)
    else:
        sns.barplot(
            x=summary.index.astype(str),
            y=summary["rate"],
            hue=summary.index.astype(str),
            palette="RdYlGn_r",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel(xlabel or column)
        ax.set_ylabel("Attrition Rate (%)")
        plt.xticks(rotation=45, ha="right")

    ax.axhline(df["Attrition_Flag"].mean() * 100, color="gray", linestyle="--", label="Company avg")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return summary


def plot_ordinal_attrition(
    df: pd.DataFrame,
    column: str,
    title: str,
    labels: dict | None = None,
    save_path=None,
) -> pd.DataFrame:
    """Attrition rate by ordinal satisfaction/level score."""
    summary = attrition_rate_table(df, column)
    if labels:
        summary.index = summary.index.map(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=summary.index.astype(str),
        y=summary["rate"],
        hue=summary.index.astype(str),
        palette="RdYlGn_r",
        legend=False,
        ax=ax,
    )
    ax.axhline(df["Attrition_Flag"].mean() * 100, color="gray", linestyle="--", label="Company avg")
    ax.set_xlabel(column)
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return summary
