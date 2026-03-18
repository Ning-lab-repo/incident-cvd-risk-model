import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from multiprocessing import Pool
import multiprocessing
from multiprocessing import shared_memory
from datetime import datetime
import warnings
import itertools
import os
import gc
import traceback
import re

warnings.filterwarnings('ignore')

# ---------------------------
# Configuration / constants
# ---------------------------
DIAG_COL = "new_diagnosis_after_baseline"
DIAG_TIME_COL = "time_new_diagnosis_after_baseline"
DEATH_ICD_COL = "newp_s_alldead"  
DEATH_DATE_COL = "new2025516_dead_data"
BASELINE_DATE_COL = "date_attending_assessment_centre"
CENSOR_DATE = pd.to_datetime("2024-07-08")


BASELINE_HISTORY_DIAG_COL = 'zhen_need_diagnosis'
BASELINE_HISTORY_TIME_COL = 'zhen_ten_need_time'


outcomes = [
    'I10', 'I25', 'I48', 'I20', 'I50', 'I51', 'I44', 'I21', 'I67', 'I08', 'I73', 'I63', 'I26', 'I45', 'I35', 'I49', 'I34', 'I27', 'I47', 'I71', 'G45', 'I24', 'I70', 'I46',
    'I69', 'I31', 'I77', 'I42', 'I65', 'I12', 'I07', 'I61', 'I74', 'I64', 'I72', 'I62', 'I78', 'I60', 'I37', 'I05', 'I36', 'I38', 'I33', 'I22', 'I11', 'I13', 'I15', 'I68'
]
# outcomes = ['I27']

# 协变量
continuous_covs = ['sample_age_days', 'BMI', 'Age', 'TDI', 'Fasting_time']
discrete_covs = ['season_binary', 'ethnicity', 'Alcohol_intake_frequency_delphi', 'Current_tobacco_smoking_delphi', 'Sex']
all_covs = continuous_covs + discrete_covs

# 进程间共享
global_covariates_df = None       # DataFrame with covariates (already imputed)
global_survival_map = None        # dict outcome -> DataFrame(duration,event)
global_protein_shm_name = None    # shared memory name
global_protein_shape = None       # (n_rows, n_proteins)
global_protein_dtype = None
global_protein_cols = None        # list of original protein names
global_protein_name_to_idx = None
global_protein_safe_map = None    # original_name -> safe_name mapping

# ---------------------------
# Utility: safe variable name for formulas
# ---------------------------
def make_safe_name(name: str) -> str:
    s = re.sub(r'\W+', '_', str(name))
    s = re.sub(r'_+', '_', s).strip('_')
    if s == '':
        s = 'var'
    if re.match(r'^\d', s):
        s = 'V_' + s
    return s

# ---------------------------
# Imputation utilities
# ---------------------------
def find_site_column(df):
    candidates = ['UK Biobank assessment centre | Instance 0']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def impute_data(df):
    df = df.copy()
    # 连续变量：中位数
    for cov in ['sample_age_days', 'Age', 'Fasting_time']:
        if cov in df.columns:
            med = df[cov].median()
            df[cov] = df[cov].fillna(med)
    # BMI：按性别中位数
    if 'BMI' in df.columns:
        if 'Sex' in df.columns:
            bmi_medians = df.groupby('Sex')['BMI'].median().to_dict()
            overall_median = df['BMI'].median()
            df['BMI'] = df.apply(lambda r: (bmi_medians.get(r['Sex'], overall_median) if pd.isnull(r['BMI']) else r['BMI']), axis=1)
        else:
            df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    # TDI：按中心中位数
    if 'TDI' in df.columns:
        site_col = find_site_column(df)
        if site_col is not None:
            site_meds = df.groupby(site_col)['TDI'].median().to_dict()
            overall = df['TDI'].median()
            df['TDI'] = df.apply(lambda r: (site_meds.get(r.get(site_col), overall) if pd.isnull(r['TDI']) else r['TDI']), axis=1)
        else:
            df['TDI'] = df['TDI'].fillna(df['TDI'].median())
    # 种族：缺失填 1（白人）
    if 'ethnicity' in df.columns:
        df['ethnicity'] = df['ethnicity'].fillna(1)
    # 离散变量：众数
    for cov in discrete_covs:
        if cov not in df.columns:
            continue
        if df[cov].isnull().any():
            try:
                m = df[cov].mode(dropna=True)
                mode_val = m.iloc[0] if len(m) > 0 else df[cov].dropna().iloc[0]
            except Exception:
                mode_val = df[cov].dropna().iloc[0]
            df[cov] = df[cov].fillna(mode_val)
    return df

# ---------------------------
# 基线 CVD 排除（全局规则）
# ---------------------------
EXCLUDED_I_PREFIXES = [
    'I79', 'I80', 'I81', 'I82', 'I83', 'I84', 'I85', 'I86', 'I87', 'I88', 'I89',
    'I95', 'I97', 'I98', 'I99'
]

def _is_cvd_code_for_baseline(code: str) -> bool:
    if not isinstance(code, str) or code == '':
        return False
    c = code.strip()
    if c.startswith('G45'):
        return True
    if c.startswith('I'):
        for p in EXCLUDED_I_PREFIXES:
            if c.startswith(p):
                return False
        return True
    return False

def compute_baseline_cvd_mask(df: pd.DataFrame) -> pd.Series:
    if BASELINE_DATE_COL not in df.columns or BASELINE_HISTORY_DIAG_COL not in df.columns or BASELINE_HISTORY_TIME_COL not in df.columns:
        print("[WARN] 缺少基线诊断/时间或基线日期列，跳过基线CVD排除。")
        return pd.Series(False, index=df.index)
    base_dates = pd.to_datetime(df[BASELINE_DATE_COL], errors='coerce', dayfirst=False)
    def row_has_baseline_cvd(r):
        base = base_dates.loc[r.name]
        if pd.isnull(base):
            return False
        try:
            codes = [s.strip() for s in str(r[BASELINE_HISTORY_DIAG_COL]).split('|') if s.strip() != '']
        except Exception:
            codes = []
        try:
            times_raw = [s.strip() for s in str(r[BASELINE_HISTORY_TIME_COL]).split('|') if s.strip() != '']
        except Exception:
            times_raw = []
        if not codes or not times_raw:
            return False
        times = pd.to_datetime(times_raw, errors='coerce', dayfirst=False)
        n = min(len(codes), len(times))
        for i in range(n):
            if pd.notnull(times[i]) and times[i] <= base and _is_cvd_code_for_baseline(codes[i]):
                return True
        return False
    return df.apply(row_has_baseline_cvd, axis=1)

# ---------------------------
# 生存构建（发病：仅诊断作为事件；死亡仅用于截尾）
# ---------------------------
def compute_survival_data_incident(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    df_local = df.copy()
    df_local[BASELINE_DATE_COL] = pd.to_datetime(df_local[BASELINE_DATE_COL], errors='coerce', dayfirst=False)

    def _row(r):
        base = r[BASELINE_DATE_COL]
        if pd.isnull(base):
            return pd.Series({'duration': np.nan, 'event': 0})

        censor_dur = (CENSOR_DATE - base).days
        if censor_dur < 0:
            return pd.Series({'duration': np.nan, 'event': 0})

        # 候选事件：匹配 outcome 的诊断最早时间
        event_days = None
        diag_col = r.get(DIAG_COL, None)
        diag_time_col = r.get(DIAG_TIME_COL, None)
        if pd.notnull(diag_col) and pd.notnull(diag_time_col):
            try:
                diags = [d.strip() for d in str(diag_col).split('|')]
                times_raw = [t.strip() for t in str(diag_time_col).split('|')]
                times_parsed = pd.to_datetime(times_raw, errors='coerce')
                n = min(len(diags), len(times_parsed))
                for i in range(n):
                    d = diags[i]
                    t = times_parsed[i]
                    if d.startswith(outcome) and pd.notnull(t):
                        dur = (t - base).days
                        if dur > 0:
                            if event_days is None or dur < event_days:
                                event_days = dur
            except Exception:
                pass

        # 死亡时间仅用于截尾
        death_days = None
        death_date_raw = r.get(DEATH_DATE_COL, None)
        if pd.notnull(death_date_raw):
            try:
                death_date = pd.to_datetime(death_date_raw, errors='coerce')
                if pd.notnull(death_date):
                    dd = (death_date - base).days
                    if dd > 0:
                        death_days = dd
            except Exception:
                pass

        # 选择结局
        if event_days is not None and event_days <= (death_days if death_days is not None else np.inf) and event_days <= censor_dur:
            return pd.Series({'duration': int(event_days), 'event': 1})
        # 否则按最早的死亡或总体截尾时间截尾
        chosen_censor = min([x for x in [death_days, censor_dur] if x is not None]) if death_days is not None else censor_dur
        return pd.Series({'duration': int(chosen_censor), 'event': 0})

    surv = df_local.apply(_row, axis=1)
    surv.index = df.index
    return surv[['duration', 'event']]

# ---------------------------
# 共享内存
# ---------------------------
def create_shared_mem_for_proteins(proteins_df):
    arr = proteins_df.astype(np.float32).values
    shape = arr.shape
    dtype = arr.dtype
    nbytes = arr.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    shm_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    return shm, shape, dtype

# ---------------------------
# Cox 拟合
# ---------------------------
def fit_cox_model_wrapper(params):
    protein_orig, outcome = params
    try:
        idx = global_protein_name_to_idx.get(protein_orig, None)
        if idx is None:
            return None

        shm = shared_memory.SharedMemory(name=global_protein_shm_name)
        prot_arr = np.ndarray(global_protein_shape, dtype=global_protein_dtype, buffer=shm.buf)
        exposure_col = prot_arr[:, idx]
        exposure_series = pd.Series(exposure_col, index=global_covariates_df.index, name=protein_orig)

        safe_name = global_protein_safe_map[protein_orig]
        required_covs = [c for c in all_covs if c in global_covariates_df.columns]
        model_df = global_covariates_df[required_covs].copy()
        model_df[safe_name] = exposure_series.values
        survival_df = global_survival_map[outcome]
        model_df = pd.concat([model_df, survival_df], axis=1)

        model_df = model_df.dropna()
        if model_df['event'].sum() < 5:
            shm.close()
            return None

        cph = CoxPHFitter()
        cov_terms = []
        for cov in required_covs:
            if cov in discrete_covs:
                cov_terms.append(f"C({cov})")
            else:
                cov_terms.append(cov)
        formula = f"{safe_name} + {' + '.join(cov_terms)}" if cov_terms else f"{safe_name}"
        cph.fit(model_df, duration_col='duration', event_col='event', formula=formula)

        try:
            summary_row = cph.summary.loc[safe_name]
        except Exception:
            mask = cph.summary.index.to_series().str.contains(re.escape(safe_name))
            if mask.any():
                summary_row = cph.summary[mask].iloc[0]
            else:
                mask2 = cph.summary.index.to_series().str.contains(re.escape(protein_orig))
                if mask2.any():
                    summary_row = cph.summary[mask2].iloc[0]
                else:
                    shm.close()
                    return None

        hr = summary_row['exp(coef)']
        hr_lower = summary_row['exp(coef) lower 95%']
        hr_upper = summary_row['exp(coef) upper 95%']
        p_value = summary_row['p']
        events = int(model_df['event'].sum())
        c_index = float(getattr(cph, 'concordance_index_', np.nan))

        shm.close()
        del model_df, exposure_series, exposure_col
        gc.collect()

        return {
            'Exposure': protein_orig,
            'Outcome': outcome,
            'Events': events,
            'Hazard ratio (95%CI)': f"{hr:.2f} ({hr_lower:.2f}-{hr_upper:.2f})",
            'P value': p_value,
            'C_index': c_index
        }

    except Exception as e:
        with open('cox_errors.log', 'a', encoding='utf-8') as f:
            f.write(f"Error for {protein_orig} x {outcome}: {str(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n---\n")
        gc.collect()
        return None

# ---------------------------
# Worker 初始化
# ---------------------------
def init_worker(covariates_df, survival_map, shm_name, shape, dtype, protein_cols, name_to_idx, safe_map):
    global global_covariates_df, global_survival_map, global_protein_shm_name, global_protein_shape, global_protein_dtype, global_protein_cols, global_protein_name_to_idx, global_protein_safe_map
    global_covariates_df = covariates_df
    global_survival_map = survival_map
    global_protein_shm_name = shm_name
    global_protein_shape = shape
    global_protein_dtype = dtype
    global_protein_cols = protein_cols
    global_protein_name_to_idx = name_to_idx
    global_protein_safe_map = safe_map

# ---------------------------
# 批处理
# ---------------------------
def process_batch(batch_params, batch_num, total_batches, n_jobs=None):
    print(f"Processing batch {batch_num}/{total_batches} with {len(batch_params)} tasks")
    if n_jobs is None:
        n_jobs = max(1, int(multiprocessing.cpu_count() * 0.5))
    with Pool(processes=n_jobs, initializer=init_worker, initargs=(global_covariates_df, global_survival_map, global_protein_shm_name, global_protein_shape, global_protein_dtype, global_protein_cols, global_protein_name_to_idx, global_protein_safe_map)) as pool:
        results = pool.map(fit_cox_model_wrapper, batch_params)
    gc.collect()
    return [r for r in results if r is not None]

# ---------------------------
# Main
# ---------------------------
def main(data_path):
    global global_protein_shm_name, global_protein_shape, global_protein_dtype, global_protein_cols, global_protein_name_to_idx, global_covariates_df, global_survival_map, global_protein_safe_map

    print("Loading data...")
    df = pd.read_csv(data_path, sep='\t' if data_path.endswith('.tsv') else ',', low_memory=False)

    # 1) 排除基线 CVD（基于 zhen_* 列）
    print("检查并排除基线 CVD 个体……")
    baseline_cvd_mask = compute_baseline_cvd_mask(df)
    n_total = len(df)
    n_baseline_cvd = int(baseline_cvd_mask.sum())
    if n_baseline_cvd > 0:
        df = df.loc[~baseline_cvd_mask].copy()
    print(f"  基线CVD: {n_baseline_cvd} / {n_total}，保留 {len(df)}")

    # 2) 选择蛋白列（2..2921）
    protein_cols = list(df.columns[1:2921])
    print(f"Using {len(protein_cols)} protein columns (columns 2..2921).")

    # 3) 安全变量名映射
    safe_map = {p: make_safe_name(p) for p in protein_cols}
    inv = {}
    for i, p in enumerate(protein_cols):
        s = safe_map[p]
        if s in inv:
            safe_map[p] = f"{s}_{i}"
        inv[safe_map[p]] = p

    # 4) 协变量与插补
    covariate_cols_present = [c for c in all_covs if c in df.columns]
    covariates_df = df[covariate_cols_present].copy()
    covariates_df.index = df.index
    print("Imputing covariates...")
    extra_cols = [c for c in [BASELINE_DATE_COL, DEATH_DATE_COL] if c in df.columns]
    helper_for_impute = pd.concat([covariates_df, df[extra_cols].copy()], axis=1)
    helper_imputed = impute_data(helper_for_impute)
    covariates_df = helper_imputed[covariate_cols_present]

    # 5) 预计算生存：仅诊断触发事件，死亡用于截尾
    print("Precomputing incident survival data for each outcome (death as censoring)...")
    survival_map = {}
    surv_cols = [c for c in [BASELINE_DATE_COL, DIAG_COL, DIAG_TIME_COL, DEATH_DATE_COL] if c in df.columns]
    df_for_surv = df[surv_cols].copy()
    for oc in outcomes:
        surv = compute_survival_data_incident(df_for_surv, oc)
        survival_map[oc] = surv
        print(f"  Outcome {oc}: incident events={int(surv['event'].sum())}")

    # 6) 蛋白共享内存
    proteins_df = df[protein_cols].copy()
    proteins_df = proteins_df.apply(pd.to_numeric, errors='coerce')
    print("Creating shared memory for protein matrix...")
    shm, shape, dtype = create_shared_mem_for_proteins(proteins_df)
    shm_name = shm.name
    print(f"  Shared memory name: {shm_name}, shape: {shape}, dtype: {dtype}")

    name_to_idx = {name: i for i, name in enumerate(protein_cols)}

    params = list(itertools.product(protein_cols, outcomes))
    total_tests = len(params)
    print(f"Total tests: {total_tests}")

    batch_size = 1000
    total_batches = (len(params) + batch_size - 1) // batch_size
    all_results = []

    # set globals
    global_protein_shm_name = shm_name
    global_protein_shape = shape
    global_protein_dtype = dtype
    global_protein_cols = protein_cols
    global_protein_name_to_idx = name_to_idx
    global_protein_safe_map = safe_map
    global_covariates_df = covariates_df
    global_survival_map = survival_map

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(params))
        batch_params = params[start_idx:end_idx]
        batch_results = process_batch(batch_params, i + 1, total_batches)
        all_results.extend(batch_results)
        print(f"Completed batch {i + 1}/{total_batches}. Results so far: {len(all_results)}")
        gc.collect()

    print("Releasing shared memory...")
    shm.close()
    shm.unlink()

    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        print("No results obtained. Check data and parameters.")
        return

    result_df['p_bonferroni'] = result_df['P value'] * total_tests
    result_df['p_bonferroni'] = result_df['p_bonferroni'].clip(upper=1.0)
    result_df = result_df[['Exposure', 'Outcome', 'Events', 'Hazard ratio (95%CI)', 'P value', 'p_bonferroni', 'C_index']]
    out_file = 'cox_incident_results.csv'
    result_df.to_csv(out_file, index=False)
    print(f"Analysis completed. Results saved to {out_file}")
    print(f"Significant results after Bonferroni correction: {(result_df['p_bonferroni'] < 0.05).sum()}")

if __name__ == '__main__':
    # 设置输入路径
    data_path = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/pro53013_morbidity_delphi.csv"
    main(data_path)

