#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "table2_predictions_leadtime01_36cvd_totalcvd_wenxianCVD_nofixedseed.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR
DEFAULT_OUTPUT_STEM = "roc_leadtime0001_top5_auc_wenxianCVD_nofixedseed_5CVD"

DEFAULT_LEAD_TIME = 0.001
DEFAULT_MODEL_A = "Disease-only Delphi"
DEFAULT_MODEL_B = "Disease+protein Delphi"
TOTAL_CVD_NAME = "Total CVD"

MODEL_COLORS = {
    "Disease-only Delphi": "#1f77b4",
    "Disease+protein Delphi": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick top-N outcomes by Disease+protein AUC and save one ROC PDF per outcome, plus Total CVD."
    )
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_stem", type=str, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--lead_time", type=float, default=DEFAULT_LEAD_TIME)
    parser.add_argument("--model_a", type=str, default=DEFAULT_MODEL_A)
    parser.add_argument("--model_b", type=str, default=DEFAULT_MODEL_B)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--include_total_cvd", action="store_true", default=True)
    parser.add_argument("--total_outcome_name", type=str, default=TOTAL_CVD_NAME)
    parser.add_argument("--total_output_stem", type=str, default="roc_leadtime01_total_cvd_two_models_wenxianCVD")
    parser.add_argument("--font_size", type=float, default=18.0)
    parser.add_argument("--fig_w", type=float, default=7.2)
    parser.add_argument("--fig_h", type=float, default=6.2)
    return parser.parse_args()


def setup_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": float(font_size),
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_input(path: Path, lead_time: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    required = {"outcome", "model", "lead_time_year", "event", "pred_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input file: {sorted(missing)}")

    x = df.copy()
    x["lead_time_year"] = pd.to_numeric(x["lead_time_year"], errors="coerce")
    x["event"] = pd.to_numeric(x["event"], errors="coerce")
    x["pred_score"] = pd.to_numeric(x["pred_score"], errors="coerce")
    x["outcome"] = x["outcome"].astype(str)
    x["model"] = x["model"].astype(str)
    x = x[np.isfinite(x["lead_time_year"]) & np.isclose(x["lead_time_year"], float(lead_time), atol=1e-12)].copy()
    return x


def list_outcomes(df: pd.DataFrame) -> List[str]:
    return sorted([x for x in pd.unique(df["outcome"].astype(str)) if x != TOTAL_CVD_NAME])


def get_curve(df: pd.DataFrame, outcome: str, model_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    sub = df[(df["outcome"] == outcome) & (df["model"] == model_name)].copy()
    sub = sub[np.isfinite(sub["event"]) & np.isfinite(sub["pred_score"])].copy()
    if sub.empty:
        return None
    y = sub["event"].astype(int).to_numpy()
    s = sub["pred_score"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    auc_val = float(auc(fpr, tpr))
    return fpr, tpr, auc_val


def pick_top_outcomes(df: pd.DataFrame, protein_model: str, top_n: int) -> List[str]:
    rows = []
    for outcome in list_outcomes(df):
        curve = get_curve(df, outcome, protein_model)
        if curve is None:
            continue
        rows.append((outcome, curve[2]))

    if not rows:
        raise RuntimeError(f"No valid ROC curves found for model={protein_model}.")

    rank = pd.DataFrame(rows, columns=["outcome", "auc_protein"])
    rank = rank.sort_values(["auc_protein", "outcome"], ascending=[False, True]).reset_index(drop=True)
    chosen = rank.head(int(top_n))
    if chosen.empty:
        raise RuntimeError(f"top_n={top_n} produced no outcomes.")
    return chosen["outcome"].tolist()


def plot_single_outcome(
    df: pd.DataFrame,
    outcome: str,
    model_a: str,
    model_b: str,
    fig_w: float,
    fig_h: float,
    font_size: float,
    out_pdf: Path,
) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    auc_values: Dict[str, float] = {}

    for model_name in [model_a, model_b]:
        curve = get_curve(df, outcome, model_name)
        if curve is None:
            continue
        fpr, tpr, auc_val = curve
        auc_values[model_name] = auc_val
        ax.plot(
            fpr,
            tpr,
            lw=3.6,
            color=MODEL_COLORS.get(model_name, None),
            label=f"{model_name} (AUC={auc_val:.3f})",
        )

    if not auc_values:
        plt.close(fig)
        raise RuntimeError(f"No valid curves for outcome={outcome}.")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", lw=1.1, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(alpha=0.22, linewidth=0.55)
    ax.set_xlabel("False-positive rate", fontsize=font_size, color="black")
    ax.set_ylabel("True-positive rate", fontsize=font_size, color="black")
    ax.set_title(outcome, fontsize=font_size + 1, color="black")
    ax.tick_params(axis="both", labelsize=max(12.0, font_size - 2), colors="black")
    ax.legend(loc="lower right", fontsize=max(11.0, font_size - 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return auc_values


def main() -> None:
    args = parse_args()
    setup_style(args.font_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(args.input_csv, args.lead_time)
    if df.empty:
        raise RuntimeError(f"No rows found after filtering lead_time={args.lead_time}.")

    top_outcomes = pick_top_outcomes(df, args.model_b, args.top_n)

    print("[Info] Top outcomes by Disease+protein AUC:")
    for idx, oc in enumerate(top_outcomes, start=1):
        out_pdf = args.output_dir / f"{args.output_stem}_{oc}.pdf"
        auc_values = plot_single_outcome(
            df=df,
            outcome=oc,
            model_a=args.model_a,
            model_b=args.model_b,
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            font_size=float(args.font_size),
            out_pdf=out_pdf,
        )
        auc_a = auc_values.get(args.model_a, float("nan"))
        auc_b = auc_values.get(args.model_b, float("nan"))
        print(f"  {idx}. {oc}: AUC_only={auc_a:.3f}, AUC_protein={auc_b:.3f}")
        print(f"     Saved: {out_pdf}")

    if args.include_total_cvd:
        total_outcome = str(args.total_outcome_name)
        out_pdf_total = args.output_dir / f"{args.total_output_stem}.pdf"
        auc_values_total = plot_single_outcome(
            df=df,
            outcome=total_outcome,
            model_a=args.model_a,
            model_b=args.model_b,
            fig_w=float(args.fig_w),
            fig_h=float(args.fig_h),
            font_size=float(args.font_size),
            out_pdf=out_pdf_total,
        )
        auc_a_total = auc_values_total.get(args.model_a, float("nan"))
        auc_b_total = auc_values_total.get(args.model_b, float("nan"))
        print(f"[Info] Total CVD: AUC_only={auc_a_total:.3f}, AUC_protein={auc_b_total:.3f}")
        print(f"       Saved: {out_pdf_total}")


if __name__ == "__main__":
    main()
