from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "auc.csv"
OUTPUT_PDF = HERE / "auc_grouped_by_icd10_chapter_boxplot.pdf"
SKIP_CHAPTERS = {"Technical", "Sex", "Smoking, Alcohol and BMI"}


def read_auc_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, skiprows=1)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", skiprows=1)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    ok = values.notna() & weights.notna()
    if ok.any() and weights[ok].sum() > 0:
        return float((values[ok] * weights[ok]).sum() / weights[ok].sum())
    return float(values.mean())


def disease_level_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (disease, chapter), group in df.groupby(["disease", "ICD-10 Chapter (short)"], sort=False):
        rows.append(
            {
                "disease": disease,
                "ICD-10 Chapter (short)": chapter,
                "mean_auc": weighted_mean(group["auc"], group["n_diseased"]),
            }
        )
    return pd.DataFrame(rows)


def load_chapter_colors(chapters: list[str]) -> dict[str, str]:
    color_map = {}
    for filename in ("df_both_13pro.csv", "df_both_13pro_cvd_death.csv"):
        path = HERE / filename
        if not path.exists():
            continue
        color_df = pd.read_csv(path)
        if {"ICD-10 Chapter (short)", "color"}.issubset(color_df.columns):
            for chapter, color in color_df[["ICD-10 Chapter (short)", "color"]].dropna().drop_duplicates().values:
                color_map.setdefault(chapter, color)

    fallback = plt.get_cmap("tab20")
    for index, chapter in enumerate(chapters):
        color_map.setdefault(chapter, fallback(index % fallback.N))
    return color_map


def main() -> None:
    df = read_auc_csv(INPUT_CSV)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["n_diseased"] = pd.to_numeric(df["n_diseased"], errors="coerce")
    df = df.dropna(subset=["auc", "disease", "ICD-10 Chapter (short)"])

    filtered_df = disease_level_auc(df).dropna(subset=["mean_auc"])
    filtered_df = filtered_df[~filtered_df["ICD-10 Chapter (short)"].isin(SKIP_CHAPTERS)]

    chapters = filtered_df["ICD-10 Chapter (short)"].drop_duplicates().tolist()
    chapter_data = {
        chapter: filtered_df.loc[filtered_df["ICD-10 Chapter (short)"] == chapter, "mean_auc"].values
        for chapter in chapters
    }
    chapter_colors = load_chapter_colors(chapters)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(1, len(chapter_data) + 1)

    for index, (chapter, values) in enumerate(chapter_data.items()):
        color = chapter_colors[chapter]
        ax.boxplot(
            values,
            positions=[positions[index]],
            patch_artist=True,
            widths=0.6,
            whis=[2.5, 97.5],
            showfliers=True,
            boxprops={"linewidth": 1.25, "facecolor": color, "edgecolor": color},
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
    ax.set_xticklabels(chapters, rotation=45, ha="right")
    ax.set_ylim(0, 1.025)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.75)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel("AUC")
    ax.set_xlabel("ICD-10 chapter")
    ax.set_title("AUC, grouped by ICD-10 chapter", y=1.05)
    plt.grid(axis="x", visible=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
