from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "df_auc_unpooled_13pro.csv"
OUTPUT_PDF = HERE / "auc_grouped_by_sex_boxplot.pdf"

NORMAL_MALE = "#0FB8A1"
NORMAL_FEMALE = "#8520F1"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    ok = values.notna() & weights.notna()
    if ok.any() and weights[ok].sum() > 0:
        return float((values[ok] * weights[ok]).sum() / weights[ok].sum())
    return float(values.mean())


def disease_sex_level_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (disease, sex), group in df.groupby(["name", "sex"], sort=False):
        rows.append(
            {
                "disease": disease,
                "sex": str(sex).lower(),
                "mean_auc": weighted_mean(group["auc"], group["n_diseased"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["n_diseased"] = pd.to_numeric(df["n_diseased"], errors="coerce")
    df = df.dropna(subset=["auc", "name", "sex"])

    filtered_df = disease_sex_level_auc(df).dropna(subset=["mean_auc"])
    female_data = filtered_df.loc[filtered_df["sex"] == "female", "mean_auc"].values
    male_data = filtered_df.loc[filtered_df["sex"] == "male", "mean_auc"].values

    fig, ax = plt.subplots(figsize=(1.75, 4))
    positions = [1, 2]
    sex_data = [female_data, male_data]
    sex_colors = [NORMAL_FEMALE, NORMAL_MALE]
    sex_labels = ["Female", "Male"]

    for index in range(2):
        ax.boxplot(
            sex_data[index],
            positions=[positions[index]],
            patch_artist=True,
            widths=0.6,
            whis=[2.5, 97.5],
            showfliers=True,
            boxprops={
                "linewidth": 1.25,
                "facecolor": sex_colors[index],
                "edgecolor": sex_colors[index],
            },
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "gray", "linewidth": 1},
            capprops={"color": "gray", "linewidth": 1},
            flierprops={
                "marker": "x",
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "markersize": 3,
                "alpha": 0.3,
            },
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(sex_labels)
    ax.set_ylim(0, 1.025)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.75)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("AUC")
    ax.set_title("AUC, grouped by sex", y=1.05)
    plt.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
