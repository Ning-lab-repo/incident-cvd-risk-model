from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "auc.csv"
OUTPUT_PDF = HERE / "auc_vs_number_tokens_training.pdf"


def read_auc_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, skiprows=1)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", skiprows=1)


def weighted_mean_auc(group: pd.DataFrame) -> float:
    weights = pd.to_numeric(group["n_diseased"], errors="coerce").fillna(0)
    values = pd.to_numeric(group["auc"], errors="coerce")
    ok = values.notna() & weights.notna()
    if ok.any() and weights[ok].sum() > 0:
        return float((values[ok] * weights[ok]).sum() / weights[ok].sum())
    return float(values.mean())


def main() -> None:
    df = read_auc_csv(INPUT_CSV)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["n_diseased"] = pd.to_numeric(df["n_diseased"], errors="coerce")
    df = df.dropna(subset=["auc", "disease", "n_diseased", "ICD-10 Chapter (short)"])

    plot_df = (
        df.groupby(["disease", "ICD-10 Chapter (short)"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "mean_auc": weighted_mean_auc(group),
                    "N tokens, training": group["n_diseased"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    plot_df = plot_df[plot_df["N tokens, training"] > 0].copy()

    chapters = sorted(plot_df["ICD-10 Chapter (short)"].dropna().unique())
    cmap = plt.get_cmap("tab20")
    color_map = {chapter: cmap(i % cmap.N) for i, chapter in enumerate(chapters)}
    colors = plot_df["ICD-10 Chapter (short)"].map(color_map)

    plt.figure(figsize=(7, 4))
    plt.scatter(
        plot_df["N tokens, training"],
        plot_df["mean_auc"],
        c=colors,
        s=24,
        edgecolor="white",
        linewidth=0.65,
    )
    plt.axhline(0.5, color="k", linestyle="--", linewidth=0.75)
    plt.title("AUC vs number of tokens in training set")
    plt.xscale("log")
    plt.ylim(0, 1.05)
    plt.xlabel("Number of tokens in training set")
    plt.ylabel("AUC")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
