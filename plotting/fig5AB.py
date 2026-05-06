import os
import textwrap
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from model import DelphiConfig, Delphi
from tqdm import tqdm
from utils import get_batch, get_p2i

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": ":",
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "font.family": "Arial",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.labelcolor": "black",
    }
)
plt.rcParams["figure.dpi"] = 72
plt.rcParams["pdf.fonttype"] = 42

FIG_SAVE_DIR = "figures_pdf"
os.makedirs(FIG_SAVE_DIR, exist_ok=True)

MIN_WITH_DISEASE_REAL = int(os.getenv("DELPHI_MIN_WITH_DISEASE_REAL", "30"))
MIN_WITHOUT_DISEASE_REAL = int(os.getenv("DELPHI_MIN_WITHOUT_DISEASE_REAL", "300"))
MIN_VALID_POINTS_FOR_CORR = 3
MAX_FALLBACK_TOKENS = int(os.getenv("DELPHI_MAX_FALLBACK_TOKENS", "6"))
MIN_EVENTS_MAIN = int(os.getenv("DELPHI_MIN_EVENTS_MAIN", "20"))
CAL_EPS = float(os.getenv("DELPHI_CAL_EPS", "1e-8"))
KEEP_STATIC_AFTER_CUTOFF = os.getenv("DELPHI_KEEP_STATIC_AFTER_CUTOFF", "1") == "1"
STATIC_CHAPTERS = set(
    filter(
        None,
        os.getenv("DELPHI_STATIC_CHAPTERS", "Sex|Smoking, Alcohol and BMI|Proteomics").split("|"),
    )
)


def save_current_figure(filename):
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    out = os.path.join(FIG_SAVE_DIR, filename)
    fig.savefig(out, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {out}")


def load_trajectory_arrays(oo):
    max_len = max(x[1].shape[1] for x in oo)
    token_arrays = [
        np.pad(x[0].cpu(), ((0, 0), (0, max_len - x[0].shape[1])), constant_values=0)
        for x in oo
    ]
    age_arrays = [
        np.pad(x[1].cpu(), ((0, 0), (0, max_len - x[1].shape[1])), constant_values=-10000)
        for x in oo
    ]
    tokens = np.concatenate(token_arrays, axis=0)
    ages = np.concatenate(age_arrays, axis=0) / 365.25
    ages = np.nan_to_num(ages, nan=-27).astype("int")
    return tokens, ages


def incidence_counts(tokens, ages, n_labels):
    out = np.zeros((n_labels, 80))
    for t in range(80):
        mask = ages == t
        out[:, t] = np.bincount(tokens[mask], minlength=n_labels).astype("float")
    return out


def survival_denominator(n, ages, death_shift=0):
    return n - np.histogram(ages.max(1) + death_shift, np.arange(81))[0].cumsum()


def resolve_requested_to_token_ids(items, labels_df, valid_ids):
    out = []
    name_series = labels_df["name"].astype(str)
    for item in items:
        resolved = None
        if len(item) >= 2 and item[0].isalpha() and item[1:].isdigit():
            match = labels_df[name_series.str.startswith(item + " ", na=False)]
            if not match.empty:
                resolved = int(match.iloc[0]["index"]) if "index" in match.columns else int(match.index[0])
        elif item.isdigit():
            if len(item) == 3 and item.startswith("1"):
                icd_guess = "I" + item[1:]
                match = labels_df[name_series.str.startswith(icd_guess + " ", na=False)]
                if not match.empty:
                    resolved = int(match.iloc[0]["index"]) if "index" in match.columns else int(match.index[0])
            if resolved is None:
                tid = int(item)
                if tid in valid_ids:
                    resolved = tid
                else:
                    icd_guess = "I" + item.lstrip("0")
                    match = labels_df[name_series.str.startswith(icd_guess + " ", na=False)]
                    if not match.empty:
                        resolved = int(match.iloc[0]["index"]) if "index" in match.columns else int(match.index[0])
        if resolved is not None and resolved in valid_ids:
            out.append(resolved)
        else:
            print(f"[WARN] Cannot resolve requested disease token: {item}")

    uniq = []
    for token_id in out:
        if token_id not in uniq:
            uniq.append(token_id)
    return uniq


def m(x, w):
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def plot_fold_change_comparison(
    fold_change_syn,
    fold_change_real,
    fold_change_syn_sd,
    fold_change_real_sd,
    labels,
    delphi_labels,
    n_plots=4,
    figsize=(14, 3),
):
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    for i in range(n_plots):
        non_nan = (
            ~np.isnan(fold_change_syn[i])
            * ~np.isnan(fold_change_real[i])
            * ~np.isinf(fold_change_syn[i])
            * ~np.isinf(fold_change_real[i])
            * (fold_change_syn[i] > 0)
            * (fold_change_real[i] > 0)
            * (np.arange(len(fold_change_real[i])) > 13)
        )

        valid_count = int(non_nan.sum())
        if valid_count >= MIN_VALID_POINTS_FOR_CORR:
            corr_weights = (1 / fold_change_syn_sd[i]) ** 2
            cw_sum = corr_weights[non_nan].sum()
            if cw_sum > 0:
                corr_weights = corr_weights / cw_sum
                r_weighted = corr(
                    np.log10(fold_change_syn[i][non_nan]),
                    np.log10(fold_change_real[i][non_nan]),
                    corr_weights[non_nan],
                )
            else:
                r_weighted = np.nan
            r = np.corrcoef(
                np.log10(fold_change_syn[i][non_nan]),
                np.log10(fold_change_real[i][non_nan]),
            )[0, 1]
        else:
            r_weighted = np.nan
            r = np.nan

        ax = axes[i]
        sizes = delphi_labels["count"].values[non_nan] / 250
        ax.scatter(
            fold_change_syn[i][non_nan],
            fold_change_real[i][non_nan],
            marker=".",
            c=delphi_labels["color"][non_nan],
            s=sizes,
        )
        ax.set_title(f"{labels[i]}, \nr={r:.2f},\nr_weighted={r_weighted:.2f}", fontsize=10)
        ax.set_xlabel("Fold change, modelled")
        if i == 0:
            ax.set_ylabel("Fold change, observed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([0.1, 30], [0.1, 30], c="k", ls=":")
        ax.set_xlim(0.1, 30)
        ax.set_ylim(0.1, 30)
        ax.set_aspect("equal")

    legend_sizes = [100, 1000, 10000]
    legend_elements = [plt.scatter([], [], c="gray", s=size / 250, label=f"n={size}") for size in legend_sizes]
    ax.legend(handles=legend_elements, title="Sample size", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


labels_path = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/delphi_labels_chapters_colours_icd_custom_13pro.csv"
delphi_labels = pd.read_csv(labels_path)

if "index" in delphi_labels.columns:
    labels_by_id = delphi_labels.set_index("index", drop=False)
else:
    labels_by_id = delphi_labels.copy()
    labels_by_id["index"] = labels_by_id.index
    labels_by_id = labels_by_id.set_index("index", drop=False)

token_id_to_name = labels_by_id["name"].astype(str).to_dict()
death_matches = labels_by_id[labels_by_id["name"].astype(str) == "Death"]
death_token_id = int(death_matches.index[0]) if not death_matches.empty else None
if death_token_id is None:
    raise ValueError("Death token not found in labels")

chapter_col = "ICD-10 Chapter (short)" if "ICD-10 Chapter (short)" in labels_by_id.columns else "ICD-10 Chapter"
chapter_series = labels_by_id[chapter_col].astype(str) if chapter_col in labels_by_id.columns else pd.Series("", index=labels_by_id.index)
static_token_mask = chapter_series.isin(STATIC_CHAPTERS)
static_token_ids = set(labels_by_id[static_token_mask].index.astype(int).tolist())
static_token_ids.discard(0)
static_token_ids.discard(1)

out_dir = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/Delphi_Myrun_13pro"
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint["model_args"])
model = Delphi(conf)
model.load_state_dict(checkpoint["model"])
model.eval()
model = model.to(device)

val = np.fromfile(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/data/ukb_simulated_data_13pro/val.bin",
    dtype=np.uint32,
).reshape(-1, 3)
val_p2i = get_p2i(val)

d = get_batch(
    range(0, val_p2i.shape[0] - 1, 1),
    val,
    val_p2i,
    select="left",
    block_size=63,
    device=device,
    padding="random",
)

age = 60
n_samples = len(d[0])
cohort_idx = np.where(
    (d[1].cpu().detach().numpy() <= age * 365.25).any(1)
    * (d[3].cpu().detach().numpy() >= age * 365.25).any(1)
)
u = np.unique(cohort_idx[0])

d0 = d[0][u[:n_samples]].clone().detach()
d1 = d[1][u[:n_samples]].clone().detach()

if KEEP_STATIC_AFTER_CUTOFF and len(static_token_ids) > 0:
    static_ids_t = torch.tensor(sorted(static_token_ids), device=d0.device, dtype=d0.dtype)
    static_after_cut = torch.isin(d0, static_ids_t) & (d1 > age * 365.25)
    d1[static_after_cut] = age * 365.25 - 1.0

d0[d1 > age * 365.25] = 0
d1[d1 > age * 365.25] = -10000.0

if age > 0:
    d0 = torch.nn.functional.pad(d0, (0, 1), "constant", 1)
    d1 = torch.nn.functional.pad(d1, (0, 1), "constant", age * 365.25)

order = d1.argsort(1)
d0 = d0.gather(1, order)
d1 = d1.gather(1, order)

batch_size = 128
n_repeats = int(os.getenv("DELPHI_N_REPEATS", "3"))
oo = []
model.to(device)

for _ in range(n_repeats):
    with torch.no_grad():
        batches = zip(*map(lambda x: torch.split(x, batch_size), (d0, d1)))
        for dd in tqdm(batches, total=len(d0) // batch_size + 1):
            generated = model.generate(
                dd[0].to(device),
                dd[1].to(device),
                max_age=80 * 365.25,
                no_repeat=True,
                max_new_tokens=128,
                termination_tokens=[death_token_id],
            )
            oo.append((generated[0], generated[1]))

syn_all_a, syn_all_b = load_trajectory_arrays(oo)
syn_inc = incidence_counts(syn_all_a, syn_all_b, len(delphi_labels))
syn_counts = syn_inc.copy()
syn_inc /= survival_denominator(len(syn_all_a), syn_all_b)

real_all_a = d[2][u[:n_samples]].cpu().detach().numpy()
real_all_b = np.nan_to_num(d[3].cpu().detach().numpy().copy()[u[:n_samples]] / 365.25, nan=-27).astype("int")
real_inc = incidence_counts(real_all_a, real_all_b, len(delphi_labels))
real_counts = real_inc.copy()
real_inc /= survival_denominator(len(real_all_a), real_all_b, death_shift=1)

ages_of_interest = [70, 71, 72, 73, 74]
sim_rate = syn_inc[:, ages_of_interest].mean(-1)
obs_rate = real_inc[:, ages_of_interest].mean(-1)
sim_events = syn_counts[:, ages_of_interest].sum(-1)
obs_events = real_counts[:, ages_of_interest].sum(-1)

valid_main = (
    (np.arange(len(delphi_labels)) > 13)
    * np.isfinite(sim_rate)
    * np.isfinite(obs_rate)
    * (sim_rate > 0)
    * (obs_rate > 0)
    * (sim_events >= MIN_EVENTS_MAIN)
    * (obs_events >= MIN_EVENTS_MAIN)
)

if valid_main.sum() == 0:
    raise ValueError("No valid tokens remain after event filtering.")

try:
    x = np.log10(sim_rate[valid_main] + CAL_EPS)
    y = np.log10(obs_rate[valid_main] + CAL_EPS)
    a_cal, b_cal = np.polyfit(x, y, 1)
except Exception as e:
    print(f"[WARN] Calibration fit failed: {e}. Fallback to identity.")
    a_cal, b_cal = 1.0, 0.0

lifestyle_tokens = [
    [7, 8, 9],
    [10, 11, 12],
    [4, 5, 6],
]

all_syn_inc_lifestyle = np.zeros((9, len(delphi_labels)))
all_real_inc_lifestyle = np.zeros((9, len(delphi_labels)))
all_syn_inc_lifestyle_token_count = np.zeros((9, len(delphi_labels)))
all_real_inc_lifestyle_token_count = np.zeros((9, len(delphi_labels)))
all_syn_inc_lifestyle_count_all = np.zeros(9)
all_real_inc_lifestyle_count_all = np.zeros(9)

for i in range(3):
    for j in range(3):
        token_of_interest = lifestyle_tokens[i][j]
        idx = 3 * i + j

        mask_syn = (syn_all_a == token_of_interest).any(1)
        syn_a_sub = syn_all_a[mask_syn]
        syn_b_sub = syn_all_b[mask_syn]
        syn_inc_sub = incidence_counts(syn_a_sub, syn_b_sub, len(delphi_labels))

        mask_real = (d[0][u[:n_samples]] == token_of_interest).any(1).cpu().numpy()
        real_a_sub = d[2][u[:n_samples]][mask_real].cpu().detach().numpy()
        real_b_sub = np.nan_to_num(
            d[3].cpu().detach().numpy().copy()[u[:n_samples]][mask_real] / 365.25,
            nan=-27,
        ).astype("int")
        real_inc_sub = incidence_counts(real_a_sub, real_b_sub, len(delphi_labels))

        all_real_inc_lifestyle_token_count[idx] = real_inc_sub[:, ages_of_interest].sum(-1)
        all_syn_inc_lifestyle_token_count[idx] = syn_inc_sub[:, ages_of_interest].sum(-1)
        syn_den = mask_syn.sum() - syn_inc_sub[int(death_token_id)].cumsum()
        real_den = mask_real.sum() - np.histogram(real_b_sub.max(1) + 1, np.arange(81))[0].cumsum()
        all_syn_inc_lifestyle_count_all[idx] = syn_den[ages_of_interest].sum(-1)
        all_real_inc_lifestyle_count_all[idx] = real_den[ages_of_interest].sum(-1)

        syn_inc_sub /= syn_den
        real_inc_sub /= real_den
        syn_inc_cal = 10 ** (a_cal * np.log10(syn_inc_sub + CAL_EPS) + b_cal)
        all_syn_inc_lifestyle[idx] = syn_inc_cal[:, ages_of_interest].mean(-1)
        all_real_inc_lifestyle[idx] = real_inc_sub[:, ages_of_interest].mean(-1)

valid_token_ids = set(delphi_labels["index"].astype(int).tolist()) if "index" in delphi_labels.columns else set(range(len(delphi_labels)))
disease_selection_raw = os.getenv("DELPHI_DISEASE_SELECTION", "I10,I25,I20,I48")
requested_items = [x.strip().upper() for x in disease_selection_raw.split(",") if x.strip()]
token_candidates = resolve_requested_to_token_ids(requested_items, delphi_labels, valid_token_ids)

if len(token_candidates) == 0:
    print("[WARN] Requested disease selection resolved to empty set; fallback to legacy defaults.")
    token_candidates = [95, 1188, 214, 305, 505, 603]

tokens = []
token_support_stats = []
fallback_candidates = []

for tok in token_candidates:
    if tok not in valid_token_ids:
        continue
    has_tok_before60 = ((real_all_a == tok) * (real_all_b < 60)).any(1)
    n_with = int(has_tok_before60.sum())
    n_without = int((~has_tok_before60).sum())
    fallback_candidates.append((tok, n_with, n_without))
    if n_with < MIN_WITH_DISEASE_REAL or n_without < MIN_WITHOUT_DISEASE_REAL:
        continue

    rb_with = real_all_b[has_tok_before60]
    rb_without = real_all_b[~has_tok_before60]
    den_with = n_with - np.histogram(rb_with.max(1) + 1, np.arange(81))[0].cumsum()
    den_without = n_without - np.histogram(rb_without.max(1) + 1, np.arange(81))[0].cumsum()
    if np.any(den_with[ages_of_interest] <= 0) or np.any(den_without[ages_of_interest] <= 0):
        continue

    tokens.append(tok)
    token_support_stats.append((tok, n_with, n_without))

if len(tokens) == 0:
    fallback_candidates = sorted(
        [x for x in fallback_candidates if x[1] > 0 and x[2] > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    if len(fallback_candidates) == 0:
        raise ValueError("No disease token has observed support before age 60.")
    selected = fallback_candidates[:MAX_FALLBACK_TOKENS]
    tokens = [x[0] for x in selected]
    token_support_stats = selected
    print("[WARN] No token passed strict support thresholds. Using fallback top tokens by n_with.")

print("Selected disease tokens (token_id, n_with, n_without):")
for tok, n_with, n_without in token_support_stats:
    print(f"  {tok}: with={n_with}, without={n_without}, name={token_id_to_name.get(int(tok), str(tok))}")

all_syn_inc_diseases = np.full((2 * len(tokens), len(delphi_labels)), np.nan)
all_real_inc_diseases = np.full((2 * len(tokens), len(delphi_labels)), np.nan)
all_syn_inc_diseases_token_count = np.zeros((2 * len(tokens), len(delphi_labels)))
all_real_inc_diseases_token_count = np.zeros((2 * len(tokens), len(delphi_labels)))
all_syn_inc_diseases_count_all = np.zeros(2 * len(tokens))
all_real_inc_diseases_count_all = np.zeros(2 * len(tokens))

for j in range(2):
    for i, token_of_interest in enumerate(tokens):
        row = j * len(tokens) + i

        if j == 0:
            mask_syn = ((syn_all_a == token_of_interest) * (syn_all_b < 60)).any(1)
            mask_real = ((real_all_a == token_of_interest) * (real_all_b < 60)).any(1)
        else:
            mask_syn = ~((syn_all_a == token_of_interest) * (syn_all_b < 60)).any(1)
            mask_real = ~((real_all_a == token_of_interest) * (real_all_b < 60)).any(1)

        syn_a_sub = syn_all_a[mask_syn]
        syn_b_sub = syn_all_b[mask_syn]
        real_a_sub = real_all_a[mask_real]
        real_b_sub = real_all_b[mask_real]

        syn_inc_sub = incidence_counts(syn_a_sub, syn_b_sub, len(delphi_labels))
        real_inc_sub = incidence_counts(real_a_sub, real_b_sub, len(delphi_labels))

        all_real_inc_diseases_token_count[row] = real_inc_sub[:, ages_of_interest].sum(-1)
        all_syn_inc_diseases_token_count[row] = syn_inc_sub[:, ages_of_interest].sum(-1)

        syn_den = mask_syn.sum() - syn_inc_sub[int(death_token_id)].cumsum()
        real_den = mask_real.sum() - np.histogram(real_b_sub.max(1) + 1, np.arange(81))[0].cumsum()

        if not (np.all(syn_den[ages_of_interest] > 0) and np.all(real_den[ages_of_interest] > 0)):
            continue

        all_syn_inc_diseases_count_all[row] = syn_den[ages_of_interest].sum()
        all_real_inc_diseases_count_all[row] = real_den[ages_of_interest].sum()

        syn_inc_sub /= syn_den
        real_inc_sub /= real_den
        syn_inc_cal = 10 ** (a_cal * np.log10(syn_inc_sub + CAL_EPS) + b_cal)
        all_syn_inc_diseases[row] = syn_inc_cal[:, ages_of_interest].mean(-1)
        all_real_inc_diseases[row] = real_inc_sub[:, ages_of_interest].mean(-1)

fold_change_real_disease = []
fold_change_syn_disease = []
fold_change_real_disease_sd = []
fold_change_syn_disease_sd = []
disease_comparisons = []
n_token_cmp = len(tokens)

for i in range(n_token_cmp):
    with np.errstate(divide="ignore", invalid="ignore"):
        fold_change_real_disease.append(all_real_inc_diseases[i] / all_real_inc_diseases[i + n_token_cmp])
        fold_change_syn_disease.append(all_syn_inc_diseases[i] / all_syn_inc_diseases[i + n_token_cmp])
        fold_change_real_disease[i][all_real_inc_diseases_token_count[i] < 10] = np.nan
        fold_change_syn_disease[i][all_syn_inc_diseases_token_count[i] < 10] = np.nan
        fold_change_real_disease_sd.append(
            np.sqrt(
                1 / all_real_inc_diseases_token_count[i]
                + 1 / all_real_inc_diseases_count_all[i]
                + 1 / all_real_inc_diseases_token_count[i + n_token_cmp]
                + 1 / all_real_inc_diseases_count_all[i + n_token_cmp]
            )
        )
        fold_change_syn_disease_sd.append(
            np.sqrt(
                1 / all_syn_inc_diseases_token_count[i]
                + 1 / all_syn_inc_diseases_count_all[i]
                + 1 / all_syn_inc_diseases_token_count[i + n_token_cmp]
                + 1 / all_syn_inc_diseases_count_all[i + n_token_cmp]
            )
        )
        disease_comparisons.append(textwrap.fill(token_id_to_name.get(int(tokens[i]), str(tokens[i])), 25))

plot_fold_change_comparison(
    fold_change_syn=fold_change_syn_disease,
    fold_change_real=fold_change_real_disease,
    fold_change_syn_sd=fold_change_syn_disease_sd,
    fold_change_real_sd=fold_change_real_disease_sd,
    labels=disease_comparisons,
    delphi_labels=delphi_labels,
    n_plots=n_token_cmp,
    figsize=(max(10, 2.4 * n_token_cmp), 4),
)
save_current_figure("05_disease_fold_change_comparison.pdf")

with np.errstate(divide="ignore", invalid="ignore"):
    fold_change_real_lifestyle = [
        all_real_inc_lifestyle[2] / all_real_inc_lifestyle[0],
        all_real_inc_lifestyle[5] / all_real_inc_lifestyle[3],
        all_real_inc_lifestyle[8] / all_real_inc_lifestyle[6],
    ]
    fold_change_syn_lifestyle = [
        all_syn_inc_lifestyle[2] / all_syn_inc_lifestyle[0],
        all_syn_inc_lifestyle[5] / all_syn_inc_lifestyle[3],
        all_syn_inc_lifestyle[8] / all_syn_inc_lifestyle[6],
    ]
    fold_change_real_lifestyle_sd = [
        np.sqrt(
            1 / all_real_inc_lifestyle_token_count[2]
            + 1 / all_real_inc_lifestyle_count_all[2]
            + 1 / all_real_inc_lifestyle_token_count[0]
            + 1 / all_real_inc_lifestyle_count_all[0]
        ),
        np.sqrt(
            1 / all_real_inc_lifestyle_token_count[5]
            + 1 / all_real_inc_lifestyle_count_all[5]
            + 1 / all_real_inc_lifestyle_token_count[3]
            + 1 / all_real_inc_lifestyle_count_all[3]
        ),
        np.sqrt(
            1 / all_real_inc_lifestyle_token_count[8]
            + 1 / all_real_inc_lifestyle_count_all[8]
            + 1 / all_real_inc_lifestyle_token_count[6]
            + 1 / all_real_inc_lifestyle_count_all[6]
        ),
    ]
    fold_change_syn_lifestyle_sd = [
        np.sqrt(
            1 / all_syn_inc_lifestyle_token_count[2]
            + 1 / all_syn_inc_lifestyle_count_all[2]
            + 1 / all_syn_inc_lifestyle_token_count[0]
            + 1 / all_syn_inc_lifestyle_count_all[0]
        ),
        np.sqrt(
            1 / all_syn_inc_lifestyle_token_count[5]
            + 1 / all_syn_inc_lifestyle_count_all[5]
            + 1 / all_syn_inc_lifestyle_token_count[3]
            + 1 / all_syn_inc_lifestyle_count_all[3]
        ),
        np.sqrt(
            1 / all_syn_inc_lifestyle_token_count[8]
            + 1 / all_syn_inc_lifestyle_count_all[8]
            + 1 / all_syn_inc_lifestyle_token_count[6]
            + 1 / all_syn_inc_lifestyle_count_all[6]
        ),
    ]

plot_fold_change_comparison(
    fold_change_syn=fold_change_syn_lifestyle,
    fold_change_real=fold_change_real_lifestyle,
    fold_change_syn_sd=fold_change_syn_lifestyle_sd,
    fold_change_real_sd=fold_change_real_lifestyle_sd,
    labels=["Smoking", "Alcohol", "BMI"],
    delphi_labels=delphi_labels,
    n_plots=3,
)
save_current_figure("06_lifestyle_fold_change_comparison.pdf")
