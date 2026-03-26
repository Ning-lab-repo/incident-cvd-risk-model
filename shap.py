# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pickle
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from model import Delphi, DelphiConfig
    from plotting import waterfall
    from utils import get_batch, get_p2i, shap_custom_tokenizer, shap_model_creator
except ImportError as exc:
    raise ImportError(
        "Failed to import Delphi modules (model/utils/plotting). "
        "Run this script in the Delphi project environment."
    ) from exc


DEFAULT_LABELS_PATH = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/"
    "delphi_labels_chapters_colours_icd_custom_13pro.csv"
)
DEFAULT_OUT_DIR = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/Delphi_Myrun_13pro"
)
DEFAULT_TRAIN_BIN = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/data/"
    "ukb_simulated_data_13pro/train.bin"
)
DEFAULT_VAL_BIN = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/data/"
    "ukb_simulated_data_13pro/val.bin"
)
DEFAULT_SHAP_AGG_PATH = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/"
    "Delphi_Myrun_13pro/shap_agg.pickle"
)
DEFAULT_SAVE_DIR = Path(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/ipynb_py_re"
)

LOCAL_DATA_DIR = SCRIPT_DIR.parent / "ukb_simulated_data_13pro"

DEFAULT_CVD_CODES = [
    "I48",
    "I27",
    "I10",
    "I65",
    "I12",
    "I50",
    "I51",
    "I42",
    "I34",
    "I25",
    "I08",
    "I73",
    "I71",
    "I44",
    "I21",
    "I36",
    "I47",
    "I49",
    "I67",
    "I20",
    "I74",
    "I70",
    "I63",
    "I35",
    "I77",
    "I69",
    "I45",
    "I46",
    "I05",
    "I26",
    "I78",
    "I37",
    "I07",
    "G45",
    "I33",
    "I31",
]

DEFAULT_PERSON = [
    ("Male", 0),
    ("No event", 40),
    ("ACTA2 high", 40),
    ("CXCL17 high", 40),
    ("MMP12 high", 40),
    ("CDCP1 high", 40),
    ("BCAN high", 40),
    ("WFDC2 high", 40),
    ("GDF15 high", 40),
    ("EDA2R high", 40),
    ("NTproBNP high", 40),
    ("CDHR2 high", 40),
    ("RBFOX3 high", 40),
    ("HAVCR1 high", 40),
    ("HSPB6 high", 40),
]

TO_EXCLUDE_PREDICTED = [
    "Technical",
    "Smoking, Alcohol and BMI",
    "Sex",
    "XVI. Perinatal Conditions",
]
TO_EXCLUDE_PREDICTOR = [
    "Technical",
    "Smoking, Alcohol and BMI",
    "Sex",
    "XVI. Perinatal Conditions",
    "Death",
]
CHAPTER_ORDER = [
    "I. Infectious Diseases",
    "II. Neoplasms",
    "III. Blood & Immune Disorders",
    "IV. Metabolic Diseases",
    "V. Mental Disorders",
    "VI. Nervous System Diseases",
    "VII. Eye Diseases",
    "VIII. Ear Diseases",
    "IX. Circulatory Diseases",
    "X. Respiratory Diseases",
    "XI. Digestive Diseases",
    "XII. Skin Diseases",
    "XIII. Musculoskeletal Diseases",
    "XIV. Genitourinary Diseases",
    "XV. Pregnancy & Childbirth",
    "XVI. Perinatal Conditions",
    "XVII. Congenital Abnormalities",
    "Death",
]


@dataclass
class RuntimeConfig:
    labels_path: Path
    out_dir: Path
    train_bin: Path
    val_bin: Path
    shap_agg_path: Path
    save_dir: Path
    device: str
    dtype: str
    seed: int
    n_min: int
    smoking_token_id: int
    time_tokens_of_interest: List[int]


def resolve_existing_path(preferred: Path, *fallbacks: Path) -> Path:
    candidates = [preferred, *fallbacks]
    for candidate in candidates:
        if candidate.exists():
            if candidate != preferred:
                print(f"[Path fallback] Using {candidate} (preferred missing: {preferred})")
            return candidate
    return preferred


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_plot_defaults() -> None:
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": ":",
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    plt.rcParams["figure.dpi"] = 72
    plt.rcParams["pdf.fonttype"] = 42


def save_pdf(fig: plt.Figure, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    if "index" not in labels.columns:
        labels = labels.copy()
        labels["index"] = labels.index.astype(int)

    labels["index"] = labels["index"].astype(int)
    labels["name"] = labels["name"].astype(str)
    return labels


def build_token_maps(labels: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_token = dict(zip(labels["index"].astype(int), labels["name"].astype(str)))
    token_to_id = {token: int(idx) for idx, token in id_to_token.items()}
    return id_to_token, token_to_id


def get_death_token_id(labels: pd.DataFrame) -> int:
    death_matches = labels[labels["name"].astype(str) == "Death"]
    if death_matches.empty:
        raise ValueError("Death token not found in labels.")
    return int(death_matches["index"].iloc[0])


def build_default_person() -> List[Tuple[str, float]]:
    return [(token, years * 365.25) for token, years in DEFAULT_PERSON]


def split_person(person: Sequence[Tuple[str, float]]) -> Tuple[List[str], List[float]]:
    tokens = [x[0] for x in person]
    ages = [x[1] for x in person]
    return tokens, ages


def tokens_to_ids(tokens: Sequence[str], token_to_id: Dict[str, int]) -> List[int]:
    return [int(token_to_id.get(t, -1)) for t in tokens]


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warn] CUDA not available, falling back to cpu")
        return "cpu"
    return requested_device


def set_seed(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_model(out_dir: Path, device: str) -> torch.nn.Module:
    ckpt_path = out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(device)
    return model


def load_train_val(train_bin: Path, val_bin: Path):
    if not train_bin.exists():
        raise FileNotFoundError(f"train.bin not found: {train_bin}")
    if not val_bin.exists():
        raise FileNotFoundError(f"val.bin not found: {val_bin}")

    train = np.fromfile(train_bin, dtype=np.uint32).reshape(-1, 3)
    val = np.fromfile(val_bin, dtype=np.uint32).reshape(-1, 3)
    train_p2i = get_p2i(train)
    val_p2i = get_p2i(val)
    return train, val, train_p2i, val_p2i


def select_diseases_of_interest(
    labels: pd.DataFrame,
    cvd_codes: Sequence[str],
    death_token_id: int,
) -> Tuple[List[int], List[str]]:
    name_series = labels["name"].astype(str)
    diseases_of_interest: List[int] = []
    missing_codes: List[str] = []

    for code in cvd_codes:
        matches = labels[name_series.str.match(rf"^{code}(\b|\s)")]
        if not matches.empty:
            diseases_of_interest.append(int(matches["index"].iloc[0]))
        else:
            missing_codes.append(code)

    diseases_of_interest.append(death_token_id)

    seen = set()
    deduped = []
    for token_id in diseases_of_interest:
        if token_id not in seen:
            deduped.append(token_id)
            seen.add(token_id)

    return deduped, missing_codes


def compute_individual_shap(
    model: torch.nn.Module,
    labels: pd.DataFrame,
    person: Sequence[Tuple[str, float]],
    id_to_token: Dict[int, str],
    token_to_id: Dict[str, int],
    death_token_id: int,
    device: str,
):
    import shap

    diseases_of_interest, missing_codes = select_diseases_of_interest(
        labels, DEFAULT_CVD_CODES, death_token_id
    )

    person_tokens, person_ages = split_person(person)
    person_tokens_ids = tokens_to_ids(person_tokens, token_to_id)

    if any(tid < 0 for tid in person_tokens_ids):
        unknown = [tok for tok, tid in zip(person_tokens, person_tokens_ids) if tid < 0]
        raise KeyError(f"Unknown tokens in person trajectory: {unknown}")

    masker = shap.maskers.Text(
        shap_custom_tokenizer,
        output_type="str",
        mask_token="10000",
        collapse_mask_token=False,
    )
    model_shap = shap_model_creator(
        model, diseases_of_interest, person_tokens_ids, person_ages, device
    )
    output_names = [id_to_token.get(i, f"TOKEN_{i}") for i in diseases_of_interest]
    explainer = shap.Explainer(model_shap, masker, output_names=output_names)

    shap_input = " ".join(str(token_to_id[t]) for t in person_tokens)
    shap_values = explainer([shap_input])
    shap_values.data = np.array(
        [[f"{token}({age / 365.25:.1f} years)" for token, age in person]]
    )

    return shap_values, person_ages, missing_codes


def save_individual_waterfall_plot(
    shap_values,
    person_ages: Sequence[float],
    save_dir: Path,
) -> None:
    with plt.style.context("default"):
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 4

        waterfall(shap_values[0, ..., 0], max_display=7, show=False, ages=person_ages)
        plt.gca().set_title("Impact of diseases on mortality", fontweight=1, size=18)
        fig = plt.gcf()

    save_pdf(fig, save_dir / "01_individual_waterfall_mortality.pdf")


def save_individual_token_bar_plot(shap_values, save_dir: Path) -> None:
    values = np.asarray(shap_values.values)[0, :, 0]
    labels = [str(x) for x in np.asarray(shap_values.data)[0]]

    order = np.argsort(np.abs(values))[::-1][:12]
    values = values[order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in values]
    ax.barh(np.arange(len(values)), values, color=colors)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-rate contribution)")
    ax.set_title("Top token contributions to mortality (single trajectory)")

    save_pdf(fig, save_dir / "02_individual_token_importance_bar.pdf")


def load_shap_aggregate(shap_agg_path: Path):
    if not shap_agg_path.exists():
        raise FileNotFoundError(f"shap_agg pickle not found: {shap_agg_path}")

    with shap_agg_path.open("rb") as f:
        shap_pkl = pickle.load(f)

    print(f"[Loaded] {shap_agg_path}")
    return shap_pkl


def build_population_df(shap_pkl) -> pd.DataFrame:
    all_tokens = np.asarray(shap_pkl["tokens"]).astype(int)
    all_values = np.asarray(shap_pkl["values"])

    df_shap = pd.DataFrame(all_values)
    df_shap["token"] = all_tokens
    return df_shap


def build_population_df_with_meta(shap_pkl) -> pd.DataFrame:
    all_tokens = np.asarray(shap_pkl["tokens"]).astype(int)
    all_values = np.asarray(shap_pkl["values"])
    all_people = np.asarray(shap_pkl["people"]).astype(int)
    all_times = np.asarray(shap_pkl["times"]).astype(float)

    n = len(all_tokens)
    if len(all_people) != n or len(all_times) != n:
        raise ValueError(
            f"shap_agg length mismatch: tokens={n}, people={len(all_people)}, times={len(all_times)}"
        )

    df_shap = pd.DataFrame(all_values)
    df_shap["token"] = all_tokens
    df_shap["person"] = all_people
    df_shap["time"] = all_times
    return df_shap


def resolve_cvd_target_tokens(
    labels: pd.DataFrame,
    cvd_codes: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    name_series = labels["name"].astype(str)
    targets = []
    missing_codes = []

    for code in cvd_codes:
        matches = labels[name_series.str.match(rf"^{code}(\b|\s)")]
        if matches.empty:
            missing_codes.append(code)
            continue

        row = matches.iloc[0]
        targets.append(
            {
                "cvd_code": code,
                "target_token_id": int(row["index"]),
                "target_name": str(row["name"]),
            }
        )

    return pd.DataFrame(targets), missing_codes


def build_person_true_next_token_map(
    person_ids: Sequence[int],
    val: np.ndarray,
    val_p2i,
) -> Dict[int, int]:
    true_next_map: Dict[int, int] = {}

    for person_id in tqdm(person_ids, desc="Building true-next token map"):
        x, a, y, b = get_batch(
            [int(person_id)],
            val,
            val_p2i,
            select="left",
            block_size=64,
            device="cpu",
            padding="random",
            cut_batch=True,
        )

        y_row = y[0]
        valid = y_row > 0
        if int(valid.sum().item()) == 0:
            true_next_map[int(person_id)] = -1
            continue

        true_next_token = int(y_row[valid][-1].item())
        true_next_map[int(person_id)] = true_next_token

    return true_next_map


def save_cvd_feature_contributions_csv(
    df_shap: pd.DataFrame,
    labels: pd.DataFrame,
    id_to_token: Dict[int, str],
    cvd_codes: Sequence[str],
    save_dir: Path,
    val: np.ndarray,
    val_p2i,
) -> None:
    target_df, missing_codes = resolve_cvd_target_tokens(labels, cvd_codes)
    if target_df.empty:
        raise ValueError("None of the target CVD codes matched labels, cannot export CSV.")

    if "person" not in df_shap.columns or "time" not in df_shap.columns:
        raise ValueError("df_shap must contain 'person' and 'time' columns.")

    # 1) keep only strict pre-prediction context tokens
    df_strict = df_shap[df_shap["time"] > 0].copy()

    # 2) exclude Death / Padding / No event from predictors
    df_strict["predictor_name"] = df_strict["token"].map(
        lambda x: id_to_token.get(int(x), f"TOKEN_{int(x)}")
    )
    excluded_predictors = {"Death", "Padding", "No event", "No_event"}
    df_strict = df_strict[~df_strict["predictor_name"].isin(excluded_predictors)].copy()

    if df_strict.empty:
        raise ValueError("No records left after filtering (time>0 and predictor exclusion).")

    # 3) strict event-based subset:
    # For each target disease, only keep samples from people whose true next event is that target.
    person_ids = sorted(df_strict["person"].astype(int).unique().tolist())
    true_next_map = build_person_true_next_token_map(person_ids, val, val_p2i)

    predictor_counts = df_strict["token"].value_counts().sort_index()
    predictor_ids = sorted(df_strict["token"].astype(int).unique().tolist())

    out_df = pd.DataFrame({"predictor_token_id": predictor_ids})
    out_df["predictor_name"] = out_df["predictor_token_id"].map(
        lambda x: id_to_token.get(int(x), f"TOKEN_{int(x)}")
    )
    out_df["sample_count"] = out_df["predictor_token_id"].map(
        lambda x: int(predictor_counts.get(int(x), 0))
    )

    n_people_per_target = []
    for _, target in target_df.iterrows():
        target_id = int(target["target_token_id"])
        code = str(target["cvd_code"])
        persons_for_target = [pid for pid, tok in true_next_map.items() if tok == target_id]
        n_people_per_target.append(len(persons_for_target))

        raw_col = f"{code}_mean_shap"
        fold_col = f"{code}_mean_fold"

        if target_id in df_strict.columns and len(persons_for_target) > 0:
            sub = df_strict[df_strict["person"].isin(persons_for_target)]
            agg = sub.groupby("token")[target_id].mean()
            out_df[raw_col] = out_df["predictor_token_id"].map(agg)
        else:
            out_df[raw_col] = np.nan
        out_df[fold_col] = np.exp(out_df[raw_col])

    out_df = out_df.reset_index(drop=True)
    target_df = target_df.copy()
    target_df["n_people_with_true_next_target"] = n_people_per_target

    ensure_dir(save_dir)
    out_csv = save_dir / "cvd36_feature_contributions_all_tokens.csv"
    mapping_csv = save_dir / "cvd36_target_mapping.csv"

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    target_df.to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {mapping_csv}")
    if missing_codes:
        print(f"[Warn] Missing CVD codes in labels: {missing_codes}")


def aggregate_population_df(
    df_shap: pd.DataFrame,
    death_token_id: int,
    n_min: int,
) -> pd.DataFrame:
    token_count_dict = df_shap["token"].value_counts().sort_index().to_dict()

    row_mask = df_shap["token"].map(token_count_dict).fillna(0) > n_min
    df_shap_agg = df_shap[row_mask].groupby("token").mean(numeric_only=True)

    if death_token_id not in df_shap_agg.columns:
        raise ValueError(
            f"death token id {death_token_id} not found in aggregated SHAP columns."
        )

    return df_shap_agg


def plot_shap_distribution(
    df_melted: pd.DataFrame,
    y_axis_labels: Sequence[str],
    group_by_col_name: str,
    title: str,
    x_lim_tuple: Tuple[float, float],
    highlight_last_dot: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3, 6), facecolor="white")

    for i, y_label in enumerate(y_axis_labels):
        data = df_melted[df_melted[group_by_col_name] == y_label]["value"]

        if len(data) == 0:
            continue

        median = np.median(data)
        quartiles = np.percentile(data, [25, 75])
        dot_color = "red" if highlight_last_dot and i == len(y_axis_labels) - 1 else "black"

        ax.plot(median, i, "o", color=dot_color, zorder=3)
        ax.hlines(i, quartiles[0], quartiles[1], color="gray", linestyles="solid", linewidth=1)

    ax.set_title(title)
    ax.set_yticks(np.arange(len(y_axis_labels)))
    ax.set_yticklabels(y_axis_labels)
    ax.tick_params(axis="x", rotation=25)
    ax.set_xscale("log")
    ax.set_xlim(*x_lim_tuple)
    ax.set_xlabel("Risk increase, folds", size=11, labelpad=10)
    return fig


def save_population_distribution_plots(
    df_shap: pd.DataFrame,
    df_shap_agg: pd.DataFrame,
    death_token_id: int,
    id_to_token: Dict[int, str],
    smoking_token_id: int,
    save_dir: Path,
) -> None:
    n_first = 20

    selected_context_tokens = df_shap_agg[death_token_id].nlargest(n_first).index[::-1]

    df_plot_source = df_shap[df_shap["token"].isin(selected_context_tokens)]
    df_plot_melted = df_plot_source[[death_token_id, "token"]].reset_index(drop=True).melt(
        id_vars=["token"], value_vars=[death_token_id]
    )
    df_plot_melted["context_token_label"] = df_plot_melted["token"].map(
        lambda x: id_to_token.get(int(x), f"TOKEN_{int(x)}")
    )
    df_plot_melted["value"] = np.exp(df_plot_melted["value"])

    y_axis_labels = [id_to_token.get(int(token), f"TOKEN_{int(token)}") for token in selected_context_tokens]
    fig = plot_shap_distribution(
        df_melted=df_plot_melted,
        y_axis_labels=y_axis_labels,
        group_by_col_name="context_token_label",
        title="Mortality factors",
        x_lim_tuple=(1, 1000),
    )
    save_pdf(fig, save_dir / "03_population_mortality_factors.pdf")

    if smoking_token_id not in df_shap_agg.index:
        print(
            f"[Skip] smoking token {smoking_token_id} not found in aggregated index, "
            "skip smoking consequence plot"
        )
        return

    shap_values_for_context = df_shap_agg.loc[smoking_token_id]
    selected_feature_tokens = shap_values_for_context.sort_values(ascending=False).index[:n_first][::-1]

    df_plot_source = df_shap[df_shap["token"] == smoking_token_id]
    selected_feature_tokens = [c for c in selected_feature_tokens if c in df_plot_source.columns]
    if not selected_feature_tokens:
        print("[Skip] no valid selected features for smoking consequence plot")
        return

    df_plot_melted = df_plot_source[[*selected_feature_tokens, "token"]].reset_index(drop=True).melt(
        id_vars=["token"],
        value_vars=selected_feature_tokens,
        var_name="feature_token_id",
        value_name="raw_shap_value",
    )
    df_plot_melted["feature_label"] = df_plot_melted["feature_token_id"].map(
        lambda x: id_to_token.get(int(x), f"TOKEN_{int(x)}")
    )
    df_plot_melted["value"] = np.exp(df_plot_melted["raw_shap_value"])

    y_axis_labels = [id_to_token.get(int(token), f"TOKEN_{int(token)}") for token in selected_feature_tokens]
    fig = plot_shap_distribution(
        df_melted=df_plot_melted,
        y_axis_labels=y_axis_labels,
        group_by_col_name="feature_label",
        title="Consequences of\nsmoking heavily",
        x_lim_tuple=(1, 11),
    )
    save_pdf(fig, save_dir / "04_population_smoking_consequences.pdf")


def get_person_from_val(
    idx: int,
    val: np.ndarray,
    val_p2i,
    id_to_token: Dict[int, str],
    device: str,
):
    x, y, _, time = get_batch(
        [idx],
        val,
        val_p2i,
        select="left",
        block_size=64,
        device=device,
        padding="random",
        cut_batch=True,
    )

    x, y = x[y > -1], y[y > -1]
    person = []
    for token_id, date in zip(x, y):
        token_int = int(token_id.item())
        person.append((id_to_token.get(token_int, f"TOKEN_{token_int}"), float(date.item())))

    last_time = time[0][-1]
    if hasattr(last_time, "item"):
        last_time = float(last_time.item())
    else:
        last_time = float(last_time)

    return person, y, last_time


def build_time_resolved_df(
    shap_pkl,
    val: np.ndarray,
    val_p2i,
    id_to_token: Dict[int, str],
    token_to_id: Dict[str, int],
    device: str,
) -> pd.DataFrame:
    ages = []
    reg_times = []

    for person_id in tqdm(np.unique(shap_pkl["people"]), desc="Collecting timeline metadata"):
        person, _, last_time = get_person_from_val(
            int(person_id), val, val_p2i, id_to_token, device
        )

        person_tokens = [x[0] for x in person]
        person_token_ids = np.array(tokens_to_ids(person_tokens, token_to_id))

        reg_time_idx = np.where(np.isin(person_token_ids, np.arange(4, 13)))[0]
        reg_time = person[reg_time_idx[0]][1] if len(reg_time_idx) > 0 else -1

        reg_times.extend([reg_time] * len(person))
        ages.extend([last_time] * len(person))

    all_tokens = np.asarray(shap_pkl["tokens"])
    all_values = np.asarray(shap_pkl["values"])
    all_times = np.asarray(shap_pkl["times"])
    all_people = np.asarray(shap_pkl["people"])

    if len(ages) != len(all_tokens):
        raise ValueError(
            f"metadata length mismatch: ages={len(ages)} vs tokens={len(all_tokens)}"
        )

    df_shap = pd.DataFrame(all_values)
    df_shap["token"] = all_tokens.astype(int)
    df_shap["time"] = all_times
    df_shap["person"] = all_people
    df_shap["age"] = np.asarray(ages) / 365.25
    df_shap["reg_time_years"] = np.asarray(reg_times) / 365.25
    df_shap["Time, years"] = df_shap["time"] / 365.25
    df_shap["age_at_token"] = df_shap["age"] - df_shap["time"] / 365.25

    df_shap = df_shap[df_shap["reg_time_years"] > 0]
    return df_shap


def bins_avg(x, y, grid_size: int = 3):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        return np.asarray([]), np.asarray([])

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if np.isclose(xmin, xmax):
        return np.asarray([xmin]), np.asarray([np.nanmean(y)])

    bin_edges = np.arange(xmin, xmax + grid_size, grid_size)
    bin_indices = np.digitize(x, bin_edges)

    bin_avgs = []
    centers = []
    for i in range(1, len(bin_edges) + 1):
        y_bin = y[bin_indices == i]
        if len(y_bin) == 0:
            continue
        x_bin = x[bin_indices == i]
        centers.append(float(np.mean(x_bin)))
        bin_avgs.append(float(np.mean(y_bin)))

    return np.asarray(centers), np.asarray(bin_avgs)


def save_time_resolved_plots(
    df_shap: pd.DataFrame,
    death_token_id: int,
    id_to_token: Dict[int, str],
    save_dir: Path,
    tokens_of_interest: Sequence[int],
) -> None:
    tokens = [int(t) for t in tokens_of_interest if int(t) in df_shap.columns]
    if not tokens:
        print("[Skip] no time-resolved tokens found in SHAP columns")
        return

    n_groups = int(np.ceil(len(tokens) / 5))
    palette_faint = [
        sns.color_palette("Paired")[0],
        sns.color_palette("Paired")[2],
        sns.color_palette("Paired")[4],
    ]
    palette_bright = [
        sns.color_palette("Paired")[1],
        sns.color_palette("Paired")[3],
        sns.color_palette("Paired")[5],
    ]

    for group_idx, token_group in enumerate(np.array_split(tokens, n_groups), start=1):
        fig, axs = plt.subplots(1, 5, figsize=(12, 2), sharey=True)
        axes = np.asarray(axs).ravel()

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(token_group):
                ax.axis("off")
                continue

            token_id = int(token_group[ax_idx])
            df_trait = df_shap[df_shap["token"] == token_id].copy()
            if death_token_id not in df_trait.columns or len(df_trait) < 2:
                ax.set_title(f"No data for {token_id}", size=8)
                ax.axis("off")
                continue

            df_trait[death_token_id] = np.exp(df_trait[death_token_id].values)
            df_trait["Time, years"] = df_trait["time"] / 365.25
            df_trait = df_trait.head(2000)

            sns.scatterplot(
                data=df_trait,
                x="Time, years",
                y=death_token_id,
                ax=ax,
                color=palette_faint[0],
                alpha=0.7,
                rasterized=True,
            )

            x_vals, y_vals = df_trait["Time, years"], df_trait[death_token_id]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_smooth, y_smooth = bins_avg(x_vals, y_vals, grid_size=3)

            if len(x_smooth) > 0:
                ax.plot(x_smooth, y_smooth, color=palette_bright[0], linewidth=1.5)

            ax.set_ylim(0.5, 500)
            ax.set_xlim(0.1, 10)
            ax.set_yscale("log")
            ax.set_ylabel("Impact on mortality")
            ax.set_title(textwrap.fill(id_to_token.get(token_id, str(token_id)), width=15), size=9)

        fig.suptitle("Time-resolved effects on mortality", y=1.04)
        save_pdf(fig, save_dir / f"05_time_resolved_group_{group_idx:02d}.pdf")


def build_time_window_aggregates(
    df_shap: pd.DataFrame,
    labels: pd.DataFrame,
    death_token_id: int,
    n_min: int,
):
    token_count_dict_below_5y = (
        df_shap[df_shap["Time, years"] < 5]["token"].value_counts().sort_index().to_dict()
    )
    token_count_dict_over_10y = (
        df_shap[df_shap["Time, years"] > 10]["token"].value_counts().sort_index().to_dict()
    )

    max_token_id = int(labels["index"].max())
    for count_dict in (token_count_dict_below_5y, token_count_dict_over_10y):
        for i in range(max_token_id + 1):
            if i not in count_dict:
                count_dict[i] = 0

    feature_cols = [c for c in df_shap.columns if isinstance(c, (int, np.integer))]
    columns_more_n = [
        int(c)
        for c in feature_cols
        if c == death_token_id
        or (
            token_count_dict_below_5y.get(int(c), 0) >= n_min
            and token_count_dict_over_10y.get(int(c), 0) >= n_min
        )
    ]

    df_below_5y = (
        df_shap[df_shap["token"].isin(columns_more_n) & (df_shap["Time, years"] < 5)]
        .groupby("token")
        .mean(numeric_only=True)
    )
    df_over_10y = (
        df_shap[df_shap["token"].isin(columns_more_n) & (df_shap["Time, years"] > 10)]
        .groupby("token")
        .mean(numeric_only=True)
    )

    df_below_5y = df_below_5y[[c for c in columns_more_n if c in df_below_5y.columns]]
    df_over_10y = df_over_10y[[c for c in columns_more_n if c in df_over_10y.columns]]

    return df_below_5y, df_over_10y


def get_tick_coords(arr: np.ndarray) -> np.ndarray:
    if len(arr) <= 1:
        return np.asarray([])
    return np.where(arr[1:] != arr[:-1])[0]


def plot_full_shap_heatmap(
    cur_df: pd.DataFrame,
    labels: pd.DataFrame,
    death_token_id: int,
    title: str,
    out_path: Path,
) -> None:
    if cur_df.empty:
        print(f"[Skip] empty heatmap input: {title}")
        return

    if death_token_id not in cur_df.columns:
        raise ValueError(f"death token id {death_token_id} missing from heatmap columns")

    cur_df = cur_df.copy()

    labels = labels.copy()
    labels = labels.set_index("index", drop=False)

    new_death_rows = 10
    max_candidates = [int(labels.index.max())]
    if len(cur_df.index) > 0:
        max_candidates.append(int(np.max(cur_df.index.values)))
    if len(cur_df.columns) > 0:
        max_candidates.append(int(np.max(cur_df.columns.values)))

    start_id = max(max_candidates) + 1
    death_clone_ids = list(range(start_id, start_id + new_death_rows))

    for c in death_clone_ids:
        cur_df[c] = cur_df[death_token_id]

    death_row = labels.loc[[death_token_id]].copy()
    death_df = pd.concat([death_row] * new_death_rows, ignore_index=True)
    death_df["name"] = [f"Death_clone_{i}" for i in range(new_death_rows)]
    death_df["index"] = death_clone_ids
    death_df.index = pd.Index(death_clone_ids)
    labels = pd.concat([labels, death_df], axis=0)

    def chapter_short(token_id: int) -> str:
        v = labels.loc[token_id, "ICD-10 Chapter (short)"]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        return str(v)

    def chapter_rank(chapter: str) -> int:
        return CHAPTER_ORDER.index(chapter) if chapter in CHAPTER_ORDER else len(CHAPTER_ORDER)

    pred_ids = [
        tid
        for tid in labels.index
        if (tid in cur_df.columns) and (chapter_short(int(tid)) not in TO_EXCLUDE_PREDICTED)
    ]
    pred_ids = sorted(pred_ids, key=lambda x: (chapter_rank(chapter_short(int(x))), int(x)))

    predictor_ids = [
        tid
        for tid in labels.index
        if (tid in cur_df.index) and (chapter_short(int(tid)) not in TO_EXCLUDE_PREDICTOR)
    ]
    predictor_ids = sorted(
        predictor_ids,
        key=lambda x: (chapter_rank(chapter_short(int(x))), int(x)),
    )

    if not pred_ids or not predictor_ids:
        print(f"[Skip] heatmap has empty axis after filtering: {title}")
        return

    cur_df = cur_df.loc[predictor_ids, pred_ids]

    row_meta = labels.loc[cur_df.index]
    col_meta = labels.loc[cur_df.columns]
    row_colors = row_meta["color"].to_numpy()
    col_colors = col_meta["color"].to_numpy()

    y_tick_coords = get_tick_coords(row_colors)
    x_tick_coords = get_tick_coords(col_colors)

    g = sns.clustermap(
        np.exp(cur_df.values),
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        col_colors=col_colors,
        norm=LogNorm(vmin=1e-1, vmax=1e1),
        cmap="RdBu_r",
        figsize=(8.5, 8.5),
        rasterized=True,
    )

    g.ax_heatmap.set_xticks(x_tick_coords)
    g.ax_heatmap.set_yticks(y_tick_coords)
    g.ax_heatmap.tick_params(length=0, width=0.5, labelsize=8)
    g.ax_cbar.tick_params(length=0.5, width=0.6, labelsize=8)

    for ch, color in labels[["ICD-10 Chapter (short)", "color"]].drop_duplicates("color").values:
        col_loc = np.where(col_colors == color)[0].mean() if (col_colors == color).any() else np.nan
        row_loc = np.where(row_colors == color)[0].mean() if (row_colors == color).any() else np.nan

        if not np.isnan(col_loc):
            g.ax_heatmap.text(col_loc - 10, -60, ch, va="bottom", rotation=90, ha="left", size=8)
        if not np.isnan(row_loc):
            g.ax_heatmap.text(-70, row_loc, ch, va="center", ha="right", size=9)

    from matplotlib.patches import Patch

    chapter_color_map = labels[["ICD-10 Chapter (short)", "color"]].drop_duplicates("color")
    chapter_color_map = chapter_color_map[
        ~chapter_color_map["ICD-10 Chapter (short)"].isin(TO_EXCLUDE_PREDICTED)
    ]
    handles = [Patch(facecolor=color) for color in chapter_color_map["color"]]
    g.ax_heatmap.legend(
        handles,
        chapter_color_map["ICD-10 Chapter (short)"],
        title="ICD-10 Chapters",
        bbox_to_anchor=(1.25, 1.0),
        loc="upper left",
        frameon=False,
    )

    g.fig.suptitle(title, y=1.02, size=10)
    save_pdf(g.fig, out_path)


def run_pipeline(cfg: RuntimeConfig) -> None:
    ensure_dir(cfg.save_dir)
    configure_plot_defaults()

    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)

    print("[1/6] Loading labels")
    labels = load_labels(cfg.labels_path)
    id_to_token, token_to_id = build_token_maps(labels)
    death_token_id = get_death_token_id(labels)
    print(f"[Info] death_token_id = {death_token_id}")

    print("[2/6] Loading model")
    model = load_model(cfg.out_dir, device)

    print("[3/6] Loading train/val data")
    _, val, _, val_p2i = load_train_val(cfg.train_bin, cfg.val_bin)

    print("[4/6] Running individual SHAP analysis")
    person = build_default_person()
    shap_values, person_ages, missing_codes = compute_individual_shap(
        model=model,
        labels=labels,
        person=person,
        id_to_token=id_to_token,
        token_to_id=token_to_id,
        death_token_id=death_token_id,
        device=device,
    )
    if missing_codes:
        print(f"[Info] Missing ICD codes in labels: {missing_codes}")

    save_individual_waterfall_plot(shap_values, person_ages, cfg.save_dir)
    save_individual_token_bar_plot(shap_values, cfg.save_dir)

    print("[5/6] Loading aggregate SHAP results and population plots")
    shap_pkl = load_shap_aggregate(cfg.shap_agg_path)
    df_shap = build_population_df(shap_pkl)
    df_shap_meta = build_population_df_with_meta(shap_pkl)
    save_cvd_feature_contributions_csv(
        df_shap=df_shap_meta,
        labels=labels,
        id_to_token=id_to_token,
        cvd_codes=DEFAULT_CVD_CODES,
        save_dir=cfg.save_dir,
        val=val,
        val_p2i=val_p2i,
    )
    df_shap_agg = aggregate_population_df(df_shap, death_token_id, cfg.n_min)
    save_population_distribution_plots(
        df_shap=df_shap,
        df_shap_agg=df_shap_agg,
        death_token_id=death_token_id,
        id_to_token=id_to_token,
        smoking_token_id=cfg.smoking_token_id,
        save_dir=cfg.save_dir,
    )

    print("[6/6] Time-resolved analysis and interaction heatmaps")
    df_shap_time = build_time_resolved_df(
        shap_pkl=shap_pkl,
        val=val,
        val_p2i=val_p2i,
        id_to_token=id_to_token,
        token_to_id=token_to_id,
        device=device,
    )
    save_time_resolved_plots(
        df_shap=df_shap_time,
        death_token_id=death_token_id,
        id_to_token=id_to_token,
        save_dir=cfg.save_dir,
        tokens_of_interest=cfg.time_tokens_of_interest,
    )

    df_below_5y, df_over_10y = build_time_window_aggregates(
        df_shap=df_shap_time,
        labels=labels,
        death_token_id=death_token_id,
        n_min=cfg.n_min,
    )

    plot_full_shap_heatmap(
        cur_df=df_below_5y,
        labels=labels,
        death_token_id=death_token_id,
        title="Influence of tokens from below 5 years, risk increase, folds",
        out_path=cfg.save_dir / "06_heatmap_below_5y.pdf",
    )
    plot_full_shap_heatmap(
        cur_df=df_over_10y,
        labels=labels,
        death_token_id=death_token_id,
        title="Influence of tokens from above 10 years, risk increase, folds",
        out_path=cfg.save_dir / "07_heatmap_over_10y.pdf",
    )

    pdf_files = sorted(cfg.save_dir.glob("*.pdf"))
    print(f"[Done] Generated {len(pdf_files)} PDF file(s) in {cfg.save_dir}")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="Run Delphi SHAP analysis and save all figures as PDF."
    )
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-bin", type=Path, default=DEFAULT_TRAIN_BIN)
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN)
    parser.add_argument("--shap-agg-path", type=Path, default=DEFAULT_SHAP_AGG_PATH)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--smoking-token-id", type=int, default=9)
    parser.add_argument(
        "--time-tokens-of-interest",
        type=int,
        nargs="*",
        default=[46, 95, 1168, 1188, 173, 214, 305, 505, 584],
    )

    args = parser.parse_args()

    labels_path = resolve_existing_path(
        args.labels_path,
        LOCAL_DATA_DIR / "delphi_labels_chapters_colours_icd_custom_13pro.csv",
    )
    train_bin = resolve_existing_path(args.train_bin, LOCAL_DATA_DIR / "train.bin")
    val_bin = resolve_existing_path(args.val_bin, LOCAL_DATA_DIR / "val.bin")

    return RuntimeConfig(
        labels_path=labels_path,
        out_dir=args.out_dir,
        train_bin=train_bin,
        val_bin=val_bin,
        shap_agg_path=args.shap_agg_path,
        save_dir=args.save_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        n_min=args.n_min,
        smoking_token_id=args.smoking_token_id,
        time_tokens_of_interest=list(args.time_tokens_of_interest),
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
