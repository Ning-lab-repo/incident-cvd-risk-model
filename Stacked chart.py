from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_PDF = "protein_icd10_chart_largefont.pdf"

# Font settings (enlarged)
AXIS_LABEL_FONTSIZE = 24
TICK_FONTSIZE = 24
LEGEND_TITLE_FONTSIZE = 36
LEGEND_LABEL_FONTSIZE = 30


def find_input_csv(script_dir: Path) -> Path:
    direct = script_dir / "selected_proteins_by_source5.csv"
    if direct.exists():
        return direct

    matches = sorted(script_dir.parent.rglob("selected_proteins_by_source5.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError("selected_proteins_by_source5.csv not found")


def build_palette(required_n: int):
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend(list(cmap.colors))

    if required_n > len(palette):
        extra = plt.get_cmap("hsv")(np.linspace(0.0, 1.0, required_n - len(palette), endpoint=False))
        palette.extend(list(extra))

    return palette[:required_n]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_csv = find_input_csv(script_dir)

    df = pd.read_csv(input_csv)

    required_cols = {"source_outcome", "selected_protein"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    grouped = df.groupby(["source_outcome", "selected_protein"]).size().reset_index(name="count")

    disease_counts = grouped.groupby("selected_protein")["source_outcome"].nunique()
    proteins_to_keep = disease_counts[disease_counts >= 2].index
    filtered = grouped[grouped["selected_protein"].isin(proteins_to_keep)]
    if filtered.empty:
        raise RuntimeError("No proteins remain after filtering (>=2 ICD-10 categories).")

    pivot = filtered.pivot_table(
        index="selected_protein",
        columns="source_outcome",
        values="count",
        fill_value=0,
    )
    totals = pivot.sum(axis=1)
    pivot = pivot.loc[totals.sort_values(ascending=False).index]

    diseases = list(pivot.columns)
    palette = build_palette(len(diseases))
    color_map = dict(zip(diseases, palette))
    plot_colors = [color_map[d] for d in diseases]

    matplotlib.rcParams["font.family"] = "Arial"
    matplotlib.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(28, 12))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=plot_colors,
        width=0.8,
    )

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=54, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Keep axis-label size aligned with A/B, while preventing overcrowded protein ticks.
    ax.tick_params(axis="x", labelsize=18, rotation=90)
    ax.tick_params(axis="y", labelsize=48)

    legend = ax.legend(
        title="Disease",
        prop={"size": LEGEND_LABEL_FONTSIZE},
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        ncol=2,
        labelspacing=0.25,
        handletextpad=0.35,
        columnspacing=0.7,
        handlelength=0.8,
        handleheight=0.6,
        borderpad=0.2,
    )
    if legend is not None:
        legend.get_title().set_fontfamily("Arial")

    # Avoid tight_layout squeezing the bar panel when legend is very tall.
    fig.subplots_adjust(left=0.07, right=0.78, bottom=0.27, top=0.96)
    output_path = script_dir / OUTPUT_PDF
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Input CSV: {input_csv}")
    print(f"Output PDF: {output_path}")
    print(f"Proteins shown: {pivot.shape[0]}, ICD-10 categories: {pivot.shape[1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
