from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAJOR_BRANCH_ORDER = ["HF", "AF", "Stroke", "CHD", "Death"]


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def coarse_branch(x: str) -> str:
    s = str(x).strip()
    if s == "MI":
        return "CHD"
    if s == "TIA":
        return "Stroke"
    if s in {"HF", "AF", "Stroke", "CHD", "Death"}:
        return s
    return "None"


def safe_savefig(fig: plt.Figure, path: Path, dpi: int = 400) -> Path:
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        fig.savefig(alt, dpi=dpi, bbox_inches="tight")
        print(f"[Warn] File is locked, saved to alternate path: {alt}")
        return alt


def build_heatmap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    use = df.copy()
    if "model" in use.columns:
        use = use[use["model"].astype(str) == "Disease+protein Delphi"].copy()
    use["branch"] = use["branch"].map(coarse_branch)
    use = (
        use.groupby(["group_var", "group", "branch"], as_index=False)["proportion"]
        .mean()
        .copy()
    )
    use["row_label"] = (
        use["group_var"].astype(str).str.replace("group_", "", regex=False)
        + " | "
        + use["group"].astype(str)
    )
    mat = (
        use.pivot_table(
            index="row_label",
            columns="branch",
            values="proportion",
            aggfunc="mean",
            observed=False,
        )
        .reindex(columns=MAJOR_BRANCH_ORDER)
        .fillna(0.0)
    )
    row_order = sorted(
        mat.index.tolist(),
        key=lambda s: (
            s.split(" | ")[0] if " | " in s else s,
            0 if "low" in s.lower() else 1 if "high" in s.lower() else 2,
            s,
        ),
    )
    mat = mat.reindex(index=row_order)
    return mat


def plot_fig5(mat: pd.DataFrame, out_pdf: Path, out_png: Path, out_matrix_csv: Path) -> tuple[Path, Path]:
    src_arr = mat.values.astype(float)
    arr = np.rot90(src_arr, 1)
    vmax = max(0.01, float(np.nanmax(arr)))
    label_slots = max(arr.shape[1], 27)
    fig_w = max(9.8, 0.38 * label_slots) + 1.2
    fig_h = max(4.8, 0.58 * arr.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)

    x_labels = [
        s.split("|", 1)[1].strip() if "|" in s else s
        for s in mat.index.tolist()
    ]
    y_labels = list(reversed(mat.columns.tolist()))
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Protein-state group")
    ax.set_ylabel("Future branch")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = float(arr[i, j])
            txt_color = "white" if v > (0.55 * vmax) else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Proportion")
    fig.tight_layout()

    mat.to_csv(out_matrix_csv, index=True)
    png_saved = safe_savefig(fig, out_png, dpi=400)
    pdf_saved = safe_savefig(fig, out_pdf, dpi=400)
    plt.close(fig)
    return png_saved, pdf_saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fig5-style 13-protein branch heatmap from protein_group_branch_proportion.csv."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("protein_group_branch_proportion.csv"),
    )
    parser.add_argument(
        "--out_pdf",
        type=Path,
        default=Path("fig5.pdf"),
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=Path("fig5.png"),
    )
    parser.add_argument(
        "--out_matrix_csv",
        type=Path,
        default=Path("fig5_matrix.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing input csv: {args.input_csv}")

    apply_plot_style()
    src = pd.read_csv(args.input_csv)
    mat = build_heatmap_matrix(src)
    if mat.empty:
        raise RuntimeError("Heatmap matrix is empty after filtering.")

    out_png, out_pdf = plot_fig5(
        mat=mat,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
        out_matrix_csv=args.out_matrix_csv,
    )
    print(f"Input: {args.input_csv.resolve()}")
    print(f"Saved: {out_pdf.resolve()}")
    print(f"Saved: {out_png.resolve()}")
    print(f"Saved: {args.out_matrix_csv.resolve()}")


if __name__ == "__main__":
    main()
