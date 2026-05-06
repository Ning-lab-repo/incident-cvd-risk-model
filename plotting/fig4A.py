import os
import torch
from model import DelphiConfig, Delphi
from tqdm import tqdm
import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt
from utils import get_batch, get_p2i

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

light_male = "#BAEBE3"
normal_male = "#0FB8A1"
dark_male = "#00574A"
light_female = "#DEC7FF"
normal_female = "#8520F1"
dark_female = "#7A00BF"

delphi_labels = pd.read_csv(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/delphi_labels_chapters_colours_icd_custom_13pro.csv"
)

diseases_of_interest = [555, 565, 587, 560, 589, 1130]
diseases_of_interest = [i for i in diseases_of_interest if i < len(delphi_labels)]
if len(diseases_of_interest) == 0:
    diseases_of_interest = list(range(min(10, len(delphi_labels))))

out_dir = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/Delphi_Myrun_13pro"
device = "cpu"
seed = 1337

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
conf = DelphiConfig(**checkpoint["model_args"])
model = Delphi(conf)
state_dict = checkpoint["model"]
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)

train = np.fromfile(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/data/ukb_simulated_data_13pro/train.bin",
    dtype=np.uint32,
).reshape(-1, 3)
val = np.fromfile(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/data/ukb_simulated_data_13pro/val.bin",
    dtype=np.uint32,
).reshape(-1, 3)

val_p2i = get_p2i(val)
dataset_subset_size = len(val_p2i)

subset_size = dataset_subset_size
d = get_batch(
    range(subset_size),
    val,
    val_p2i,
    select="left",
    block_size=80,
    device=device,
    padding="random",
)

is_male = (d[0] == 3).any(axis=1).cpu().numpy()
has_gender = is_male | (d[0] == 2).any(axis=1).cpu().numpy()

p = []
model.to(device)
batch_size = 256
subset_size = dataset_subset_size
with torch.no_grad():
    for d_batch in tqdm(
        zip(*map(lambda x: torch.split(x, batch_size), d)),
        total=d[0].shape[0] // batch_size + 1,
    ):
        p.append(model(*d_batch)[0].cpu().detach())
p = torch.vstack(p)
d = [d_.cpu() for d_ in d]

females = train[np.isin(train[:, 0], train[train[:, 2] == 1, 0])]
males = train[np.isin(train[:, 0], train[train[:, 2] == 2, 0])]
n_females = (train[:, 2] == 1).sum()
n_males = (train[:, 2] == 2).sum()

unique_male_indices = np.where(males[:-1, 0] != males[1:, 0])[0]
unique_female_indices = np.where(females[:-1, 0] != females[1:, 0])[0]


def calc_age_distribution(data, unique_indices):
    ages = np.maximum(40, np.round(data[unique_indices, 1] / 365.25) + 1)
    counts = np.histogram(ages, np.arange(100))[0]
    cumulative = -np.cumsum(counts)
    return cumulative - cumulative[-1]


n_males = calc_age_distribution(males, unique_male_indices)
n_females = calc_age_distribution(females, unique_female_indices)


def plot_age_incidence(ix, d, p, highlight_idx=0):
    import math

    n_cols = 3
    n_rows = math.ceil(len(ix) / n_cols)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(12, 3.6 * n_rows), sharex=False, sharey=True
    )
    axf = np.array(ax).ravel()
    legend_handles = []
    legend_labels = []

    for i, k in enumerate(ix):
        cur_ax = axf[i]
        x_age = d[1][:, :].detach().numpy() / 365.25
        y_rate = np.exp(p.detach().numpy()[:, :, k]) * 365.25
        y_rate = 1 - np.exp(-y_rate)

        no_prior_disease = ~np.isin(d[0], k).any(axis=1)
        mask = has_gender * no_prior_disease
        valid_x = x_age[mask].ravel()
        valid_y = y_rate[mask].ravel()
        n_points = min(5000, len(valid_x))
        sub_sample = np.random.randint(0, len(valid_x), n_points)

        sc1 = cur_ax.scatter(
            valid_x[sub_sample],
            valid_y[sub_sample],
            marker=".",
            c=np.repeat(
                np.array([light_female, light_male])[0 + is_male[mask]], x_age.shape[1]
            ).ravel()[sub_sample],
            edgecolors="white",
            s=50,
            label="Delphi, all time steps",
            rasterized=True,
        )

        has_k = np.where(d[2].detach().numpy()[has_gender] == k)[0]
        before_k = d[2].detach().numpy()[has_gender].ravel() == k
        sc2 = cur_ax.scatter(
            x_age[has_gender].ravel()[before_k],
            y_rate[has_gender].ravel()[before_k],
            marker=".",
            c=np.array([dark_female, dark_male])[0 + is_male[has_gender][has_k]],
            edgecolors="white",
            s=50,
            label="Delphi, penultimate step before disease",
            rasterized=True,
        )

        j = np.where(np.isin(d[2], k).any(axis=1))[0][highlight_idx]
        j0 = np.where(x_age[j] >= 0)[0][0]
        jk = np.where(d[2][j, :].detach().numpy() == k)[0][0]
        line_case = cur_ax.plot(
            x_age[j][j0 : jk + 1],
            y_rate[j][j0 : jk + 1],
            ds="steps-post",
            c="k",
            ls="-",
            marker=".",
            markersize=8,
            markeredgecolor="white",
            markerfacecolor="k",
            label="selected case",
        )[0]
        cur_ax.scatter(
            x_age[j][jk], y_rate[j][jk], marker=".", s=200, edgecolors="white", c="k", zorder=3
        )

        h, bins = np.histogram(females[females[:, 2] == k - 1, 1] / 365.25, np.arange(100))
        stair_female = cur_ax.stairs(
            h / n_females, bins, color=normal_female, lw=2, label="reported incidence, female"
        )

        h, bins = np.histogram(males[males[:, 2] == k - 1, 1] / 365.25, np.arange(100))
        stair_male = cur_ax.stairs(
            h / n_males, bins, color=normal_male, lw=2, label="reported incidence, male"
        )

        if i == 0:
            legend_handles = [sc1, sc2, line_case, stair_female, stair_male]
            legend_labels = [
                "Delphi, all time steps",
                "Delphi, penultimate step before disease",
                "selected case",
                "reported incidence, female",
                "reported incidence, male",
            ]

        cur_ax.set_ylim((1e-5, 1))
        cur_ax.set_xlim((0, 80))
        cur_ax.set_yscale("log")
        cur_ax.set_title(
            "\n".join(textwrap.wrap(delphi_labels.loc[k, "name"], width=30)),
            verticalalignment="top",
            fontsize=10,
            fontweight="bold",
        )

        if i % n_cols == 0:
            cur_ax.set_ylabel("Rate per year")
        if i // n_cols == n_rows - 1:
            cur_ax.set_xlabel("Age")

    for j in range(len(ix), len(axf)):
        fig.delaxes(axf[j])

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.07),
            ncol=3,
            frameon=False,
            fontsize=12,
        )

    fig.tight_layout(rect=[0, 0.14, 1, 1], h_pad=1.0, w_pad=1.2)
    return fig


fig = plot_age_incidence(diseases_of_interest, d, p, highlight_idx=0)
plt.show()
