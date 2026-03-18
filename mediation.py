
import os, sys
import warnings

# =============================================================================
# 1. 全局并行配置 (在导入 numpy/scipy/statsmodels 之前)
# =============================================================================

# --- 1.1 命令行预解析 (来自用户的参考代码) ---
def _preparse_int(argv, keys, default):
    """
    在加载 numpy/pandas 之前，从 sys.argv 快速解析整数参数。
    """
    val = None
    for i, a in enumerate(argv):
        if a in keys and i+1 < len(argv):
            try:
                val = int(argv[i+1])
            except Exception:
                pass
    return val if val is not None else default

# --- 1.2 硬编码 *默认* 参数 ---
# 这些是基础设置，但可以被命令行的 --jobs 和 --blas-threads 覆盖
DEFAULT_JOBS = 120
DEFAULT_BLAS_THREADS = 1

# --- 1.3 执行预解析 ---
# 允许用户通过 `python script.py --jobs 80 --blas-threads 2` 覆盖
JOBS = _preparse_int(sys.argv, ("--jobs","-j"), DEFAULT_JOBS)
BLAS_THREADS = _preparse_int(sys.argv, ("--blas-threads","--blas_threads"), DEFAULT_BLAS_THREADS)

# --- 1.4 设置 BLAS 环境变量 (!!! 必须在 import numpy 之前) ---
for var in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = str(BLAS_THREADS)

# --- 1.5 定义全局 HARDCODED_CONFIG ---
# 将解析后的值存入 config, 供后续代码使用
HARDCODED_CONFIG = {
    # 并行设置 (使用预解析的值)
    "jobs": JOBS,
    "blas_threads": BLAS_THREADS,
    "chunk_size": 256, # TableA 的蛋白分块大小

    # 文件路径
    "input_file": "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/pro53013_morbidity_delphi_no_baseline_cvd.csv",
    "input_sep": ",",
    "base_out_dir": ".", # 输出文件将保存在当前目录

    # TableA (线性)
    "protein_start_col": 2,
    "protein_end_col": 2921,

    # TableC (中介)
    "sims": 1000,
    "alpha": 0.05,
    "seed": 42,
    "treat_mode": "sd",
    "treat_delta": 1.0,
    "min_events": 10,
    "na_scope": "per-mediator",
    "scale_mediator": True,
}

print(f"[Config] BLAS Threads set to: {BLAS_THREADS} (可被 --blas-threads 覆盖)")
print(f"[Config] Parallel Jobs set to: {JOBS} (可被 --jobs 覆盖)")

# 屏蔽特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# 2. 导入依赖库 (在 BLAS 设置之后)
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import bs  # noqa: F401  # 预留样条支持
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# =============================================================================
# 3. 共享常量定义
# =============================================================================

# --- 核心变量 ---
EXPOSURES = ["WC", "HBA1C"]
OUTCOMES = [
    "I48","I27","I10","I65","I12","I50","I51","I42","I34","I25","I08","I73",
    "I71","I44","I21","I36","I47","I49","I67","I20","I74","I70","I63","I35",
    "I77","I69","I45","I46","I05","I26","I78","I15","I37","I07","G45","I33","I31"
]

# --- 协变量 ---
continuous_covs = ['sample_age_days', 'BMI', 'Age', 'TDI', 'Fasting_time']
discrete_covs = ['season_binary', 'ethnicity', 'Alcohol_intake_frequency_delphi', 'Current_tobacco_smoking_delphi', 'Sex']
all_covs = continuous_covs + discrete_covs

# --- 生存分析列定义 ---
DIAG_COL = "new_diagnosis_after_baseline"
DIAG_TIME_COL = "time_new_diagnosis_after_baseline"
DEATH_ICD_COL = "newp_s_alldead"
DEATH_DATE_COL = "new2025516_dead_data"
BASELINE_DATE_COL = "date_attending_assessment_centre"
CENSOR_DATE = pd.to_datetime("2024-07-08")
EVENT_COL = "event" # TableC 中使用

# =============================================================================
# 4. 共享工具函数 (插补, 并行, 生存构建)
# =============================================================================

# --- 4.1 TQDM + Joblib ---
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

# --- 4.2 协变量插补 ---
def find_site_column(df):
    candidates = ['UK Biobank assessment centre | Instance 0']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 连续变量（普通）：全局中位数
    for cov in ['sample_age_days', 'Age', 'Fasting_time']:
        if cov in df.columns:
            med = pd.to_numeric(df[cov], errors='coerce').median()
            df[cov] = pd.to_numeric(df[cov], errors='coerce').fillna(med)
            
    # BMI：按 Sex 分组中位数
    if 'BMI' in df.columns:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
        if 'Sex' in df.columns:
            tmp = df[['BMI','Sex']].copy()
            tmp['Sex'] = pd.to_numeric(tmp['Sex'], errors='coerce')
            bmi_medians = tmp.groupby('Sex', dropna=False)['BMI'].median().to_dict()
            overall_median = tmp['BMI'].median()
            
            def _fill_bmi(row):
                v = row['BMI']
                if pd.isna(v):
                    sx = pd.to_numeric(row.get('Sex'), errors='coerce')
                    return bmi_medians.get(sx, overall_median)
                return v
            df['BMI'] = df.apply(_fill_bmi, axis=1)
        else:
            df['BMI'] = df['BMI'].fillna(df['BMI'].median())
            
    # TDI：按站点中位数
    if 'TDI' in df.columns:
        df['TDI'] = pd.to_numeric(df['TDI'], errors='coerce')
        site_col = find_site_column(df)
        if site_col is not None and site_col in df.columns:
            site_meds = df.groupby(site_col, dropna=False)['TDI'].median().to_dict()
            overall = df['TDI'].median()
            def _fill_tdi(row):
                v = row['TDI']
                if pd.isna(v):
                    return site_meds.get(row.get(site_col), overall)
                return v
            df['TDI'] = df.apply(_fill_tdi, axis=1)
        else:
            df['TDI'] = df['TDI'].fillna(df['TDI'].median())
            
    # ethnicity：缺失填 1（白人）
    if 'ethnicity' in df.columns:
        df['ethnicity'] = df['ethnicity'].fillna(1)
        
    # 其他离散变量：众数
    for cov in discrete_covs:
        if cov not in df.columns:
            continue
        if df[cov].isnull().any():
            try:
                m = df[cov].mode(dropna=True)
                mode_val = m.iloc[0] if len(m) > 0 else df[cov].dropna().iloc[0]
            except Exception:
                try:
                    mode_val = df[cov].dropna().iloc[0]
                except Exception:
                    mode_val = 0 # 最终回退
            df[cov] = df[cov].fillna(mode_val)
    return df

# --- 4.3 生存数据构建 (Table B) ---
def parse_first_date(s):
    """从 | 分隔的字符串中解析最早的有效日期"""
    if pd.isna(s) or str(s).strip() == "":
        return pd.NaT
    parts = [p.strip() for p in str(s).split("|") if p.strip() != ""]
    if not parts:
        return pd.NaT
    dates = pd.to_datetime(parts, errors="coerce", dayfirst=False, yearfirst=True, infer_datetime_format=True)
    dates = dates.dropna()
    return dates.min() if len(dates) else pd.NaT

def find_first_event_date(diag_str, time_str, target_icd):
    """在 | 分隔的诊断字符串中查找特定ICD(前缀匹配)的最早日期"""
    if pd.isna(diag_str) or pd.isna(time_str) or str(diag_str).strip() == "":
        return pd.NaT
    try:
        diags = str(diag_str).split('|')
        times = str(time_str).split('|')
        event_dates = []
        for i in range(min(len(diags), len(times))):
            diag = diags[i].strip()
            time_val = times[i].strip()
            if diag.startswith(target_icd):
                dt = pd.to_datetime(time_val, errors='coerce', dayfirst=False, yearfirst=True, infer_datetime_format=True)
                if pd.notna(dt):
                    event_dates.append(dt)
        return min(event_dates) if event_dates else pd.NaT
    except Exception:
        return pd.NaT

def build_survival(df, outcome_icd):
    """为特定的 outcome_icd 构建生存数据 (time 和 event)"""
    base_dt = pd.to_datetime(df[BASELINE_DATE_COL], errors="coerce", dayfirst=False, yearfirst=True, infer_datetime_format=True)
    death_dt = df[DEATH_DATE_COL].apply(parse_first_date)
    morbidity_dt = df.apply(
        lambda row: find_first_event_date(row.get(DIAG_COL), row.get(DIAG_TIME_COL), outcome_icd),
        axis=1
    )
    dates_df = pd.DataFrame({'morbidity': morbidity_dt, 'death': death_dt, 'censor': CENSOR_DATE})
    event_dt = dates_df.min(axis=1, skipna=True)
    status = (event_dt == morbidity_dt) & morbidity_dt.notna()
    diff = (event_dt - base_dt)
    time_days = diff.dt.total_seconds() / (24.0 * 3600.0)
    out = df.copy()
    out["time"] = time_days
    out["event"] = status.astype(int)
    out = out[out["time"].notna()]
    out = out[out["time"] > 0]
    return out

# --- 4.4 事件构建 (Table C) ---
def earliest_event_date_for_code(diag_str, time_str, code3):
    if pd.isna(diag_str) or str(diag_str).strip() == "": return pd.NaT
    if pd.isna(time_str) or str(time_str).strip() == "": return pd.NaT
    diag_tokens = [x.strip() for x in str(diag_str).split('|') if x.strip() != ""]
    time_tokens = [x.strip() for x in str(time_str).split('|') if x.strip() != ""]
    m = min(len(diag_tokens), len(time_tokens))
    best = pd.NaT
    for i in range(m):
        if diag_tokens[i][:3] == code3:
            dt = pd.to_datetime(time_tokens[i], errors='coerce')
            if pd.isna(dt): continue
            if pd.isna(best) or dt < best:
                best = dt
    return best

def build_event_for_outcome(df, code3):
    base = pd.to_datetime(df[BASELINE_DATE_COL], errors='coerce')
    evt  = df.apply(lambda r: earliest_event_date_for_code(r.get(DIAG_COL), r.get(DIAG_TIME_COL), code3), axis=1)
    event = evt.notna() & base.notna() & (evt > base) # 确保事件在基线后
    return event.astype(int)

# --- 4.5 Cox 分析 (Table B worker) ---
def run_cox_one(prot, df_base, covar_cols):
    try:
        cols = ["time", "event"] + covar_cols + [prot]
        dd = df_base[cols].copy()
        for c in covar_cols + [prot]:
            dd[c] = pd.to_numeric(dd[c], errors='coerce')
        dd = dd.dropna()
        if dd.shape[0] < 50 or dd["event"].sum() < 10:
            return [prot, np.nan, np.nan, np.nan, np.nan]
        cph = CoxPHFitter()
        cph.fit(dd, duration_col="time", event_col="event", show_progress=False)
        if prot not in cph.summary.index:
            return [prot, np.nan, np.nan, np.nan, np.nan]
        row = cph.summary.loc[prot]
        hr = float(row.get('exp(coef)', np.exp(row['coef'])))
        ci_l = float(row.get('exp(coef) lower 95%', np.exp(row['coef lower 95%'])))
        ci_u = float(row.get('exp(coef) upper 95%', np.exp(row['coef upper 95%'])))
        p = float(row['p'])
        return [prot, hr, ci_l, ci_u, p]
    except Exception:
        return [prot, np.nan, np.nan, np.nan, np.nan]

# --- 4.7 线性回归 (Table A worker) ---
def run_ols_one_parallel(prot, df_model_base, exposure_col, cov_terms, all_covs_list):
    """
    (Table A 并行单元)
    为单个蛋白质运行 OLS。
    返回: [protein, beta, se, pval]
    """
    rhs_terms = [exposure_col] + cov_terms
    formula = f"Q('{prot}') ~ " + " + ".join(rhs_terms)
    
    try:
        # 仅提取此模型所需的列，减少内存占用
        cols_to_use = [exposure_col, prot] + all_covs_list
        # 确保所有列都在 df_model_base 中，以防万一
        cols_to_use = [c for c in cols_to_use if c in df_model_base.columns]
        data = df_model_base[cols_to_use].copy()
        
        res = smf.ols(formula=formula, data=data).fit()
        beta = res.params.get(exposure_col, np.nan)
        se   = res.bse.get(exposure_col, np.nan)
        pval = res.pvalues.get(exposure_col, np.nan)
        
        return [prot, beta, se, pval]
    
    except Exception:
        # 捕获 OLS 失败
        return [prot, np.nan, np.nan, np.nan]


# --- 4.6 中介分析 (Table C worker) ---
def mvn_draws(mean, cov, n_draws, rng):
    mean = np.asarray(mean, dtype=np.float64)
    cov  = np.asarray(cov,  dtype=np.float64)
    k = mean.shape[0]
    eye = np.eye(k, dtype=np.float64)
    jitter = 0.0
    for _ in range(5):
        try:
            return rng.multivariate_normal(mean, cov + jitter*eye, size=n_draws)
        except np.linalg.LinAlgError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 10
    # 最终尝试
    try:
        return rng.multivariate_normal(mean, cov + 1e-4*eye, size=n_draws)
    except Exception as e:
        print(f"[WARN] mvn_draws 失败: {e}", file=sys.stderr)
        # 返回均值作为退路
        return np.tile(mean, (n_draws, 1))


def mediate_one_quasi(prot, dfb, exposure_col, config,
                      covar_cols):
    
    cols = [exposure_col] + covar_cols + [prot, EVENT_COL]
    cols = [c for c in cols if c in dfb.columns]
    dd = dfb.loc[:, cols].copy()
    
    if len(dd.columns) < (3 + len(covar_cols)):
        return [prot] + [np.nan]*12 + [0]

    dd = dd.dropna()
    n_eff = int(dd.shape[0])
    if n_eff == 0 or dd[EVENT_COL].sum() < config['min_events']:
        return [prot] + [np.nan]*12 + [n_eff]

    for c in covar_cols + [exposure_col, EVENT_COL, prot]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    dd[EVENT_COL] = dd[EVENT_COL].astype("int8")

    if config['scale_mediator']:
        mu = dd[prot].mean(); sd = dd[prot].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            return [prot] + [np.nan]*12 + [n_eff]
        dd[prot] = (dd[prot] - mu) / sd

    EXPO = dd[exposure_col].to_numpy(np.float64)
    COVS_NP = [dd[c].to_numpy(np.float64) for c in covar_cols if c in dd.columns]
    M = dd[prot].to_numpy(np.float64)
    Y = dd[EVENT_COL].to_numpy(np.int8)

    mu_w = EXPO.mean(); sd_w = EXPO.std(ddof=1)
    if config['treat_mode'] == "delta":
        control, treat = mu_w, mu_w + float(config['treat_delta'])
    else: # 'sd'
        control, treat = mu_w, mu_w + sd_w

    X_m = sm.add_constant(np.column_stack([EXPO] + COVS_NP), prepend=True)
    m_fit = sm.OLS(M, X_m).fit()

    X_y = sm.add_constant(np.column_stack([EXPO, M] + COVS_NP), prepend=True)
    try:
        y_fit = sm.Logit(Y, X_y).fit(disp=0)
    except Exception:
        try:
            glm = sm.GLM(Y, X_y, family=sm.families.Binomial())
            y_fit = glm.fit()
        except Exception as e_glm:
            print(f"[WARN] Logit/GLM 失败 (prot={prot}): {e_glm}", file=sys.stderr)
            return [prot] + [np.nan]*12 + [n_eff]

    beta_m = m_fit.params; beta_y = y_fit.params

    def logit_prob(params, w, m, covs_values_list):
        z = params[0] + params[1]*w + params[2]*m
        for i, cov_val_array in enumerate(covs_values_list, start=3):
            z += params[i] * cov_val_array
        return 1.0 / (1.0 + np.exp(-z))

    M_treat   = beta_m[0] + beta_m[1]*treat
    M_control = beta_m[0] + beta_m[1]*control
    for i, cov in enumerate(COVS_NP, start=2):
        M_treat   += beta_m[i] * cov
        M_control += beta_m[i] * cov

    p_c_m_treat   = logit_prob(beta_y, control, M_treat,   COVS_NP)
    p_c_m_control = logit_prob(beta_y, control, M_control, COVS_NP)
    p_t_m_control = logit_prob(beta_y, treat,   M_control, COVS_NP)
    p_t_m_treat   = logit_prob(beta_y, treat,   M_treat,   COVS_NP)

    ACME_hat = float(np.mean(p_c_m_treat - p_c_m_control))
    ADE_hat  = float(np.mean(p_t_m_control - p_c_m_control))
    TE_hat   = float(np.mean(p_t_m_treat  - p_c_m_control))
    PM_hat   = float(ACME_hat / TE_hat) if np.isfinite(TE_hat) and TE_hat != 0 else np.nan

    rng = np.random.default_rng(config['seed'] + (abs(hash(prot)) % 10_000_000))
    try:
        cov_m = m_fit.cov_params()
    except Exception:
        cov_m = np.eye(len(beta_m)) * 1e-6
    try:
        cov_y = y_fit.cov_params()
    except Exception:
        cov_y = np.eye(len(beta_y)) * 1e-6

    sims = config['sims']
    m_draws = mvn_draws(beta_m, cov_m, sims, rng)
    y_draws = mvn_draws(beta_y, cov_y, sims, rng)

    acmes, ades, pms = [], [], []
    for i in range(sims):
        bm = m_draws[i, :]; by = y_draws[i, :]
        M_treat_i   = bm[0] + bm[1]*treat
        M_control_i = bm[0] + bm[1]*control
        for j, cov in enumerate(COVS_NP, start=2):
            M_treat_i   += bm[j] * cov
            M_control_i += bm[j] * cov
        
        def lp_i(w, m, covs_values_list):
            z = by[0] + by[1]*w + by[2]*m
            for k, cov_val_array in enumerate(covs_values_list, start=3):
                z += by[k] * cov_val_array
            return 1.0 / (1.0 + np.exp(-z))
            
        acme_i = float(np.mean(lp_i(control, M_treat_i, COVS_NP) - lp_i(control, M_control_i, COVS_NP)))
        ade_i  = float(np.mean(lp_i(treat,   M_control_i, COVS_NP) - lp_i(control, M_control_i, COVS_NP)))
        te_i   = float(np.mean(lp_i(treat,   M_treat_i,   COVS_NP) - lp_i(control, M_control_i, COVS_NP)))
        pm_i   = float(acme_i / te_i) if np.isfinite(te_i) and te_i != 0 else np.nan
        acmes.append(acme_i); ades.append(ade_i); pms.append(pm_i)

    acmes = np.asarray(acmes, dtype=np.float64)
    ades  = np.asarray(ades,  dtype=np.float64)
    pms   = np.asarray(pms,   dtype=np.float64)

    def ci_two_sided(draws, alpha):
        draws = draws[np.isfinite(draws)]
        if draws.size == 0:
            return (np.nan, np.nan, np.nan)
        lo, hi = np.percentile(draws, [100*alpha/2, 100*(1-alpha/2)])
        p = 2 * min((draws >= 0).mean(), (draws <= 0).mean())
        p = float(min(max(p, 0.0), 1.0))
        return (lo, hi, p)

    alpha = config['alpha']
    acme_lo, acme_hi, acme_p = ci_two_sided(acmes, alpha)
    ade_lo,  ade_hi,  ade_p  = ci_two_sided(ades,  alpha)
    pm_lo,   pm_hi,   pm_p   = ci_two_sided(pms,   alpha)

    return [prot, ADE_hat, ade_lo, ade_hi, ade_p,
            ACME_hat, acme_lo, acme_hi, acme_p,
            PM_hat,  pm_lo,  pm_hi,  pm_p,
            n_eff]

# =============================================================================
# 5. 流程函数 (Table A, B, C)
# =============================================================================

# --------------------------- 5.1 Table A (线性回归) ---------------------------
def run_table_a(exposure_col, df_raw, config):
    """
    执行 TableA (线性回归): [Exposure] vs Proteins
    """
    print(f"\n[Table A] 开始处理: {exposure_col} vs Proteins")
    
    out_file = os.path.join(config['base_out_dir'], f"TableA_{exposure_col}_vs_Proteins.csv")
    
    # 提取蛋白列（按位置）
    pro_start = config['protein_start_col'] - 1
    pro_end = config['protein_end_col'] # 已改为闭区间，所以 -1 + 1 = 0
    protein_cols = list(df_raw.columns[pro_start:pro_end])
    p_total = len(protein_cols)
    print(f"[Table A] 蛋白列数: {p_total} (位置 {config['protein_start_col']}~{config['protein_end_col']})")
    
    # 数值转型：Exposure 与协变量
    present_covs = [c for c in all_covs if c in df_raw.columns]
    keep_cols = [exposure_col] + present_covs
    site_col = find_site_column(df_raw)
    if site_col:
        keep_cols = list(dict.fromkeys(keep_cols + [site_col]))
        
    df_cov = df_raw[keep_cols].copy()
    
    for c in keep_cols:
        df_cov[c] = pd.to_numeric(df_cov[c], errors="coerce")
        
    # 剔除 Exposure 缺失
    n_before = len(df_cov)
    df_cov = df_cov[~df_cov[exposure_col].isna()].copy()
    print(f"[Table A] 剔除 {exposure_col} 缺失: {n_before - len(df_cov)} 行, 保留 {len(df_cov)} 行")
    
    # 协变量插补
    helper = impute_data(df_cov)
    df_cov = helper[[c for c in df_cov.columns if c != site_col or site_col is None]]
    
    # 蛋白列转 numeric + 中位数填充
    print(f"[Table A] 转换 {p_total} 个蛋白质列 (to_numeric + 中位数插补)...")
    df_pro = df_raw.loc[df_cov.index, protein_cols].apply(pd.to_numeric, errors="coerce")
    medians = df_pro.median(axis=0)
    df_pro = df_pro.fillna(medians)
    
    # 组装建模 DataFrame (包含所有协变量和所有插补后的蛋白质)
    print(f"[Table A] 组装基础 DataFrame...")
    df_model_base = pd.concat([df_cov, df_pro], axis=1)
    
    # 构造协变量项
    cov_terms = []
    for c in present_covs:
        if c in discrete_covs:
            cov_terms.append(f"C({c})")
        else:
            cov_terms.append(c)
            
    # --- (!!! NEW: Parallel execution, replacing chunking) ---
    print(f"[Table A] 开始并行 OLS (n_jobs={config['jobs']}) ...")
    
    with tqdm_joblib(tqdm(total=p_total, desc=f"TableA ({exposure_col})", unit="prot")) as progress_bar:
        results = Parallel(n_jobs=config['jobs'], backend="loky")(
            delayed(run_ols_one_parallel)(
                prot, df_model_base, exposure_col, cov_terms, present_covs
            ) 
            for prot in protein_cols
        )
    
    # --- (END OF NEW SECTION) ---
            
    print(f"[Table A] 整理 {len(results)} 个结果...")
    # 从 results 列表中提取结果
    name_list = [r[0] for r in results]
    beta_list = [r[1] for r in results]
    se_list   = [r[2] for r in results]
    p_list    = [r[3] for r in results]
            
    out = pd.DataFrame({
        "protein": name_list,
        "Beta": np.array(beta_list, dtype=float),
        "SE":   np.array(se_list, dtype=float),
        "P_value": np.array(p_list, dtype=float)
    })
    
    out["P_bonferroni"] = (out["P_value"] * p_total).clip(upper=1.0)
    out["Protein_count"] = p_total
    out = out.sort_values("P_value").reset_index(drop=True)
    out.to_csv(out_file, index=False)
    
    print(f"[Table A] 输出完成 -> {out_file}")
    return out_file

# --------------------------- 5.2 Table B (Cox 回归) ---------------------------
def run_table_b_cox(exposure_col, table_a_path, df_raw, config):
    """
    执行 TableB (Cox 回归): Protein vs Outcomes
    使用 TableA 结果 (Beta > 0 & P_bonf < 0.05) 筛选蛋白
    """
    print(f"\n[Table B] 开始处理: {exposure_col} (TableA: {table_a_path})")
    
    out_dir = os.path.join(config['base_out_dir'], f"Cox_Morbidity_Results_{exposure_col}")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 筛选蛋白 (来自 TableA) ---
    try:
        tableA = pd.read_csv(table_a_path)
    except FileNotFoundError:
        print(f"[Table B Error] TableA 文件未找到: {table_a_path}", file=sys.stderr)
        return out_dir, {} # 返回空
        
    prot_col = 'protein' if 'protein' in tableA.columns else 'Protein'
    beta_col = 'Beta' if 'Beta' in tableA.columns else None
    p_bonf_col = 'P_bonferroni' if 'P_bonferroni' in tableA.columns else None

    if prot_col not in tableA.columns or beta_col is None or p_bonf_col is None:
        print(f"[Table B Error] TableA 列不全 (protein, Beta, P_bonferroni) in {table_a_path}", file=sys.stderr)
        return out_dir, {}

    tableA[beta_col] = pd.to_numeric(tableA[beta_col], errors='coerce')
    tableA[p_bonf_col] = pd.to_numeric(tableA[p_bonf_col], errors='coerce')

    filter_cond = (tableA[beta_col] > 0) & (tableA[p_bonf_col] < 0.05)
    cand_prots = (tableA.loc[filter_cond, prot_col]
                    .dropna().astype(str).unique().tolist())
    
    print(f"[Table B] 候选蛋白数 (Beta > 0 & P_bonf < 0.05): {len(cand_prots)}")
    if not cand_prots:
        print("[Table B Warning] 未筛选到任何候选蛋白，跳过此暴露。")
        return out_dir, {}

    # --- 协变量列表 (检查数据中实际存在的) ---
    all_covs_in_data = [c for c in (continuous_covs + discrete_covs) if c in df_raw.columns]
    if not all_covs_in_data:
        print("[Table B Error] 找不到任何协变量列。", file=sys.stderr)
        return out_dir, {}

    # --- 按结局循环 ---
    print(f"[Table B] 开始处理 {len(OUTCOMES)} 个结局...")
    outcome_sample_sizes = {} # 用于收集样本量

    for outcome_icd in tqdm(OUTCOMES, desc=f"TableB Outcomes ({exposure_col})", unit="outcome"):
        
        # 1. 构建生存数据
        df_surv = build_survival(df_raw.copy(), outcome_icd)

        # 收集样本量
        n_total = len(df_surv)
        n_events = df_surv['event'].sum()
        outcome_sample_sizes[outcome_icd] = {'n_total': n_total, 'n_events': n_events}
        
        if n_total == 0 or n_events < 10:
            # print(f"  [Table B Warn] 结局 {outcome_icd} 的有效事件数不足 (<10)，跳过。") # 减少噪音
            continue

        # 2. 协变量估算
        df_surv = impute_data(df_surv)

        # 3. 蛋白质估算 (中位数)
        present_prots = [p for p in cand_prots if p in df_surv.columns]
        if not present_prots:
            # print(f"  [Table B Warn] 候选蛋白在数据中均未找到，跳过 {outcome_icd}。")
            continue
            
        for p in list(present_prots): # 迭代副本，因为可能移除元素
            s = pd.to_numeric(df_surv[p], errors='coerce')
            med = np.nanmedian(s)
            if not np.isnan(med):
                df_surv[p] = np.where(np.isnan(s), med, s)
            else:
                present_prots.remove(p) # 蛋白质完全缺失

        if not present_prots:
            # print(f"  [Table B Warn] 所有候选蛋白在 {outcome_icd} 中均完全缺失，跳过。")
            continue
            
        # 4. 准备基础数据框
        keep_cols = set(["time","event"]) | set(all_covs_in_data) | set(present_prots)
        df_base = df_surv.loc[:, [c for c in df_surv.columns if c in keep_cols]].copy()

        for c in all_covs_in_data:
            df_base[c] = pd.to_numeric(df_base[c], errors='coerce')

        # 5. 并行运行 Cox
        with tqdm_joblib(tqdm(total=len(present_prots), desc=f"Cox ({outcome_icd})", unit="prot", leave=False)):
            results = Parallel(n_jobs=config['jobs'], backend="loky")(
                delayed(run_cox_one)(p, df_base, all_covs_in_data) for p in present_prots
            )

        # 6. 整理并保存结果
        out = pd.DataFrame(results, columns=["protein", "HR", "CI_low", "CI_high", "P_value"])
        out = out.dropna(subset=['P_value'])
        
        m = max(1, len(present_prots)) # 实际测试的蛋白数量
        out["P_bonf"] = (out["P_value"] * m).clip(upper=1.0)
        
        out["HR_95CI"] = out.apply(
            lambda r: f"{r.HR:.3f} ({r.CI_low:.3f}-{r.CI_high:.3f})" if pd.notna(r.HR) else "",
            axis=1
        )
        out = out.sort_values("P_value", na_position="last").reset_index(drop=True)
        
        out_filename = os.path.join(out_dir, f"{outcome_icd}_Cox_Results.csv")
        out.to_csv(out_filename, index=False)

    print(f"[Table B] {exposure_col} 所有结局处理完成。结果保存在: {out_dir}")
    return out_dir, outcome_sample_sizes

# --------------------------- 5.3 Table C (中介分析) ---------------------------
def run_table_c_mediation(exposure_col, cox_results_dir, df_raw, config):
    """
    执行 TableC (中介分析): [Exposure] -> Protein -> Outcome
    使用 TableB (Cox) 结果 (HR > 1 & P_bonf < 0.05) 筛选蛋白
    """
    print(f"\n[Table C] 开始处理: {exposure_col} (Cox Dir: {cox_results_dir})")
    
    out_dir = os.path.join(config['base_out_dir'], f"TableC_Mediation_Results_{exposure_col}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 准备基础数据 (插补协变量)
    # 注意：这里我们只插补协变量和暴露。蛋白质将在每个结局循环内部插补。
    print("[Table C] 正在插补协变量...")
    df_imputed = impute_data(df_raw.copy())
    
    # 协变量 + 暴露 数值化
    present_covs = [c for c in all_covs if c in df_imputed.columns]
    for c in present_covs + [exposure_col]:
        if c in df_imputed.columns:
            df_imputed[c] = pd.to_numeric(df_imputed[c], errors="coerce")
        else:
            print(f"[Table C Error] 关键列 {c} 不在数据中!", file=sys.stderr)
            return out_dir
            
    # --- 按结局循环 ---
    print(f"[Table C] 开始处理 {len(OUTCOMES)} 个结局的中介分析...")

    for code3 in tqdm(OUTCOMES, desc=f"TableC Outcomes ({exposure_col})", unit="outcome"):
        
        # 1. 筛选蛋白 (来自 Table B)
        coxpath = os.path.join(cox_results_dir, f"{code3}_Cox_Results.csv")
        if not os.path.exists(coxpath):
            # print(f"[Table C Warn] Cox 文件缺失: {coxpath}, 跳过 {code3}")
            continue
        try:
            cox = pd.read_csv(coxpath)
        except Exception as e:
            print(f"[Table C Warn] 读取 Cox 文件失败: {coxpath}: {e}")
            continue
            
        prot_col, p_col, hr_col = None, None, None
        for c in ["protein","Protein","prot","Prot"]:
            if c in cox.columns: prot_col = c; break
        for c in ["P_bonf","P_bonferroni","p_bonf","p_bonferroni"]:
            if c in cox.columns: p_col = c; break
        for c in ["HR","hr","exp(coef)"]:
            if c in cox.columns: hr_col = c; break
            
        if prot_col is None or p_col is None or hr_col is None:
            # print(f"[Table C Warn] Cox 列缺失（protein/HR/P_bonf），跳过 {code3}") # 减少噪音
            continue
            
        sel = cox[(pd.to_numeric(cox[hr_col], errors='coerce') > 1.0) & (pd.to_numeric(cox[p_col], errors='coerce') < 0.05)]
        cand_prots = sel[prot_col].dropna().astype(str).tolist()
        cand_prots = [p for p in cand_prots if p in df_imputed.columns]
        
        if len(cand_prots) == 0:
            # print(f"[Table C Info] 结局 {code3}: 无满足条件的蛋白，跳过。")
            continue

        # 2. 构建事件 (二分类)
        event_series = build_event_for_outcome(df_imputed, code3)
        if event_series.sum() < config['min_events']:
            # print(f"[Table C Warn] 结局 {code3}: 事件数不足 ({event_series.sum()})，跳过。") # 减少噪音
            continue

        # 3. 准备中介分析数据
        # 复制所需数据，减少内存占用
        cols_to_keep = [exposure_col] + present_covs + cand_prots
        dfb = df_imputed.loc[event_series.index, [c for c in cols_to_keep if c in df_imputed.columns]].copy()

        # 蛋白中位数填充 (仅填充候选蛋白)
        for p in cand_prots:
            s = pd.to_numeric(dfb[p], errors='coerce')
            med = np.nanmedian(s)
            if not np.isnan(med):
                dfb[p] = np.where(np.isnan(s), med, s)
            # 如果全 nan，将在 mediate_one_quasi 内部被 dropna 
            
        dfb[EVENT_COL] = event_series.values

        # 缺失范围 (per-mediator)
        # 注意: config['na_scope'] == "per-mediator" 是默认行为，
        # mediate_one_quasi 内部会自行处理 dd = dd.dropna()
        
        # 4. 并行运行中介分析
        with tqdm_joblib(tqdm(total=len(cand_prots), desc=f"{code3} Mediation", unit="prot", leave=False)):
            results = Parallel(n_jobs=config['jobs'], backend="loky")(
                delayed(mediate_one_quasi)(
                    p, dfb,
                    exposure_col=exposure_col,
                    config=config,
                    covar_cols=present_covs
                )
                for p in cand_prots
            )

        # 5. 整理输出
        cols = ["protein",
                "ADE", "ADE_low", "ADE_high", "ADE_p",
                "ACME","ACME_low","ACME_high","ACME_p",
                "PM",  "PM_low",  "PM_high",  "PM_p",
                "N"]
        out = pd.DataFrame(results, columns=cols)
        out = out.dropna(subset=['ADE_p', 'ACME_p'], how='all')
        if out.empty:
            continue

        def fmt_ci(v, lo, hi):
            if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi): return ""
            return f"{v:.4e} ({lo:.4e},{hi:.4e})"
        def fmt_pm(pm, lo, hi):
            if not np.isfinite(pm) or not np.isfinite(lo) or not np.isfinite(hi): return ""
            return f"{pm*100:.2f}% ({lo*100:.2f}%,{hi*100:.2f}%)"

        out["ADE_str"]  = out.apply(lambda r: fmt_ci(r["ADE"],  r["ADE_low"],  r["ADE_high"]), axis=1)
        out["ACME_str"] = out.apply(lambda r: fmt_ci(r["ACME"], r["ACME_low"], r["ACME_high"]), axis=1)
        out["PM_str"]   = out.apply(lambda r: fmt_pm(r["PM"],   r["PM_low"],   r["PM_high"]), axis=1)

        out = out.sort_values(["ACME_p","ADE_p"], na_position="last").reset_index(drop=True)
        out_path = os.path.join(out_dir, f"TableC_{code3}_Mediation.csv")
        out.to_csv(out_path, index=False)

    print(f"[Table C] {exposure_col} 所有结局处理完成。结果保存在: {out_dir}")
    return out_dir

# =============================================================================
# 6. 主函数 (Orchestrator)
# =============================================================================
def main():
    # config 从全局 HARDCODED_CONFIG 获取 (已包含预解析的 jobs/blas_threads)
    config = HARDCODED_CONFIG
    
    print(f"[Main] 开始完整分析流程 (共 {len(EXPOSURES)} 个暴露变量)")
    print(f"[Main] 加载主数据文件: {config['input_file']}")
    
    try:
        # 使用 dtype=str 加载，确保所有列（尤其是蛋白列和ID列）被正确读取
        df_raw = pd.read_csv(config['input_file'], sep=config['input_sep'], dtype=str)
    except FileNotFoundError:
        print(f"[Main Error] 输入文件未找到: {config['input_file']}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Main Error] 读取文件时出错: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"[Main] 数据加载完成. Shape={df_raw.shape}")
    
    all_sample_stats = {} # 收集所有暴露和结局的样本量

    # --- 循环处理每个暴露变量 ---
    for exp_col in EXPOSURES:
        print(f"\n{'='*20} STARTING PIPELINE FOR EXPOSURE: {exp_col} {'='*20}")
        
        # 检查暴露列是否存在
        if exp_col not in df_raw.columns:
            print(f"[Main Error] 暴露列 '{exp_col}' 在输入文件中未找到. 跳过此暴露。")
            continue
            
        try:
            # 步骤 1: Table A (线性回归)
            # 传入 df_raw.copy() 确保原始数据不被修改
            table_a_path = run_table_a(exp_col, df_raw.copy(), config)
            
            # 步骤 2: Table B (Cox 回归)
            cox_dir, sample_stats = run_table_b_cox(exp_col, table_a_path, df_raw.copy(), config)
            all_sample_stats[exp_col] = sample_stats
            
            # 步骤 3: Table C (中介分析)
            run_table_c_mediation(exp_col, cox_dir, df_raw.copy(), config)
            
            print(f"\n{'='*20} COMPLETED PIPELINE FOR EXPOSURE: {exp_col} {'='*20}")

        except Exception as e:
            print(f"\n[Main Error] 暴露 {exp_col} 的流程执行失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print(f"--- 自动跳过 {exp_col} 并继续下一个 ---")
            
    # --- 7. 生成样本量汇总表 ---
    print("\n[Main] 所有流程执行完毕。正在生成样本量汇总表...")
    try:
        records = []
        for exp, outcomes_data in all_sample_stats.items():
            if not outcomes_data: continue
            for outcome, stats in outcomes_data.items():
                records.append({
                    "Exposure": exp,
                    "Outcome": outcome,
                    "N_Total_Survival": stats.get('n_total', 0),
                    "N_Events_Survival": stats.get('n_events', 0)
                })
        
        if records:
            summary_df = pd.DataFrame(records)
            
            # 生成透视表以便阅读
            summary_pivot = summary_df.pivot(
                index="Outcome", 
                columns="Exposure", 
                values=["N_Total_Survival", "N_Events_Survival"]
            )
            
            # 展平多重索引
            if isinstance(summary_pivot.columns, pd.MultiIndex):
                summary_pivot.columns = [f"{val}_{exp}" for val, exp in summary_pivot.columns]
            
            summary_file = os.path.join(config['base_out_dir'], "pipeline_sample_size_summary.csv")
            summary_pivot.reset_index().to_csv(summary_file, index=False)
            print(f"[Main] 样本量汇总表已保存 -> {summary_file}")
        else:
            print("[Main] 未收集到样本量数据。")

    except Exception as e:
        print(f"[Main Error] 生成样本量汇总表失败: {e}", file=sys.stderr)

    print("\n[Main] 全部分析流程结束。")

if __name__ == "__main__":
    main()


