import os
import pandas as pd
import numpy as np
import torch
from model import DelphiConfig, Delphi
import matplotlib.pyplot as plt
from utils import shap_custom_tokenizer, shap_model_creator
import shap
from plotting import waterfall

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

delphi_labels = pd.read_csv(
    "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/delphi_labels_chapters_colours_icd_custom_13pro.csv"
)

out_dir = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/delpi/Delphi/Delphi_Myrun_13pro"
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
conf = DelphiConfig(**checkpoint["model_args"])
model = Delphi(conf)
model.load_state_dict(checkpoint["model"])
model.eval()
model = model.to(device)

person = [
    ("Female", 0),
    ("No event", 40),
    ("D05 Carcinoma in situ of breast", 42),
    ("H52 Disorders of refraction and accommodation", 46),
    ("N20 Calculus of kidney and ureter", 54),
    ("EDA2R high", 60),
    ("GDF15 high", 60),
    ("HAVCR1 high", 60),
    ("CDCP1 high", 60),
    ("H26 Other cataract", 64),
]
person = [(a, b * 365.25) for a, b in person]

if "index" in delphi_labels.columns:
    id_to_token = dict(
        zip(delphi_labels["index"].astype(int), delphi_labels["name"].astype(str))
    )
else:
    id_to_token = delphi_labels["name"].to_dict()
token_to_id = {v: int(k) for k, v in id_to_token.items()}

cvd_codes = [
    "I48",
    "I50",
    "I64",
    "I12",
    "I51",
    "I25",
    "I08",
    "I34",
    "I71",
    "I42",
    "I73",
    "I10",
    "I44",
    "I70",
    "I47",
    "I46",
    "I67",
    "I21",
    "I49",
    "I05",
    "I20",
    "I69",
    "I45",
    "I63",
    "I77",
    "I35",
    "I26",
    "I74",
    "I13",
    "I24",
    "I61",
    "I15",
    "I65",
    "I11",
    "I78",
    "I38",
    "I72",
    "I37",
    "G45",
    "I33",
    "I68",
]

name_series = delphi_labels["name"].astype(str)
diseases_of_interest = []
missing_codes = []
for code in cvd_codes:
    matches = delphi_labels[name_series.str.match(rf"^{code}(\b|\s)")]
    if not matches.empty:
        if "index" in matches.columns:
            diseases_of_interest.append(int(matches["index"].iloc[0]))
        else:
            diseases_of_interest.append(int(matches.index[0]))
    else:
        missing_codes.append(code)

death_matches = delphi_labels[name_series == "Death"]
if not death_matches.empty:
    if "index" in death_matches.columns:
        diseases_of_interest.append(int(death_matches["index"].iloc[0]))
    else:
        diseases_of_interest.append(int(death_matches.index[0]))

seen = set()
diseases_of_interest = [
    x for x in diseases_of_interest if not (x in seen or seen.add(x))
]

print(f"Selected targets: {len(diseases_of_interest)}")
if missing_codes:
    print("Missing ICD codes in labels:", missing_codes)

person_tokens = [i[0] for i in person]
person_ages = [i[1] for i in person]
person_tokens_ids = [token_to_id[t] for t in person_tokens]

masker = shap.maskers.Text(
    shap_custom_tokenizer,
    output_type="str",
    mask_token="10000",
    collapse_mask_token=False,
)
model_shap = shap_model_creator(
    model, diseases_of_interest, person_tokens_ids, person_ages, device
)
explainer = shap.Explainer(
    model_shap, masker, output_names=[id_to_token[i] for i in diseases_of_interest]
)
shap_values = explainer([" ".join(list(map(lambda x: str(token_to_id[x]), person_tokens)))])
shap_values.data = np.array(
    [
        list(
            map(
                lambda x: f"{id_to_token[token_to_id[x[0]]]}({x[1]/365:.1f} years) ",
                person,
            )
        )
    ]
)

print("\nGenerating plot 2: Impact of prior events on Hypertension risk...")
target_disease_id = 587

try:
    target_disease_index = diseases_of_interest.index(target_disease_id)
    target_disease_name = delphi_labels["name"].values[target_disease_id]

    with plt.style.context("default"):
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["figure.dpi"] = 150
        plt.rcParams["font.size"] = 8

        plt.figure(figsize=(14, 8))

        waterfall(
            shap_values[0, :, target_disease_index],
            max_display=20,
            show=False,
            ages=person_ages,
        )

        if plt.gcf().get_axes():
            ax = plt.gca()
            title = f"Impact of prior events on the risk of\n{target_disease_name}"
            ax.set_title(title, fontweight="bold", fontsize=14, pad=16)

            plt.subplots_adjust(left=0.42, right=0.95, top=0.88, bottom=0.10)

            plt.savefig(
                "hypertension_waterfall.pdf",
                format="pdf",
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

except ValueError:
    print(f"错误：目标疾病ID {target_disease_id} 不在 `diseases_of_interest` 列表中。")
    print("请确保将目标疾病ID添加到列表中，然后重新运行。")
except IndexError:
    print(f"错误: 无法找到疾病名称 ID {target_disease_id}。请检查 'delphi_labels' 文件。")
