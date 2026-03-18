# -*- coding: gbk -*-
import os
import numpy as np
import pandas as pd
import joblib  # 导入 joblib 用于保存模型
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, precision_score, recall_score
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ---------------- 基本路径与常量 ----------------
file_path = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/pro53013_morbidity_delphi_no_baseline_cvd.csv"
DIAG_COL = "new_diagnosis_after_baseline"
# DEATH_ICD_COL = "newp_s_alldead" # 移除

#base_path = '/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/model/verify/pro_pre_model' #/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/model/verify/test_selectes_pro/test/
base_path = '/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/model/verify/test_selectes_pro/test/'
selected_proteins_path = os.path.join(base_path, "selected_proteins_for_modeling5.csv") #selected_proteins_for_modelingv5.csv
if not os.path.exists(base_path):
    os.makedirs(base_path)

# 结局列表36
outcomes = ["I48","I27","I10","I65","I12","I50","I51","I42","I34","I25","I08","I73","I71","I44","I21","I36","I47","I49","I67","I20","I74","I70","I63","I35","I77","I69","I45","I46","I05","I26","I78","I37","I07","G45","I33","I31"]

# 自助法参数
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------- 载入数据与特征集 ----------------
data = pd.read_csv(file_path)

selected_proteins_df = pd.read_csv(selected_proteins_path)
selected_proteins = selected_proteins_df['selected_proteins'].tolist()

# 仅选择特征列（不做任何提前插补！）
# 为稳健起见，与数据列求交集，避免列缺失抛错
selected_proteins = [c for c in selected_proteins if c in data.columns]
if len(selected_proteins) == 0:
    raise ValueError("selected_proteins 在数据中一个也没找到，请检查列名。")

X_all = data[selected_proteins].copy()

# ---------------- 工具函数 ----------------
def format_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f"{value:.4f}"

def compute_metrics(y_true, y_pred, sample_weight=None):
    # 保证混淆矩阵形状稳定
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    sn = recall_score(y_true, y_pred, sample_weight=sample_weight)
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    # 若某折/重采样内某类未预测出，MCC 会报错或无意义，做个保护
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    f1 = f1_score(y_true, y_pred, sample_weight=sample_weight)
    return acc, sn, sp, ppv, npv, mcc, f1

def get_ci(arr, conf=CONFIDENCE_LEVEL):
    if len(arr) == 0:
        return np.nan, np.nan
    lower = np.percentile(arr, (1 - conf) / 2 * 100)
    upper = np.percentile(arr, (1 + conf) / 2 * 100)
    return lower, upper

# ---------------- 单结局处理 ----------------
def process_outcome(outcome):
    # 构造二分类标签 y (仅使用 DIAG_COL)
    diag_list = data[DIAG_COL].fillna('').astype(str).str.split('|')
    y = diag_list.apply(lambda codes: any(code.startswith(outcome) for code in codes)).astype(int)

    num_positive = int(y.sum())
    if num_positive == 0:
        return {'outcome': outcome, 'numy': 0}  # 无阳性样本则跳过

    # 评价时的样本权重：只基于 y 的全局分布（不涉及特征，不会泄露）
    class_freq = y.value_counts(normalize=True)
    weights_eval = y.map({0: 1 / class_freq.get(0, 1.0), 1: 1 / class_freq.get(1, 1.0)}).values

    # 10 折分层交叉验证（折内：拟合插补器 + 训练模型）
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    y_pred_prob = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in cv.split(X_all, y):
        X_train_raw, X_val_raw = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_train = y.iloc[train_idx]

        # 折内拟合插补器（median）
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)

        # 按训练折重算不平衡比
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos = (neg / pos) if pos > 0 else 1.0

        # 训练 XGB
        model = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.01,
            scale_pos_weight=scale_pos,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist'  # 更快更稳
        )

        model.fit(X_train, y_train)
        y_pred_prob[val_idx] = model.predict_proba(X_val)[:, 1]

    # 用所有 OOF 概率计算 ROC / AUC（不含泄露）
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Youden 最优阈值（在 OOF 上选择）
    youden = tpr - fpr
    optimal_idx = int(np.argmax(youden))
    optimal_threshold = float(thresholds[optimal_idx])
    predictions = (y_pred_prob >= optimal_threshold).astype(int)

    # 指标（可选带权）
    acc, sn, sp, ppv, npv, mcc, f1 = compute_metrics(y, predictions, sample_weight=weights_eval)

    # ------ 【新增】训练并保存最终模型（使用所有数据） ------
    # 1. 在所有数据上拟合插补器
    imputer_final = SimpleImputer(strategy='median')
    X_all_imputed = imputer_final.fit_transform(X_all)

    # 2. 计算全局的 scale_pos_weight
    pos_final = int(y.sum())
    neg_final = int(len(y) - pos_final)
    scale_pos_final = (neg_final / pos_final) if pos_final > 0 else 1.0

    # 3. 训练最终模型
    model_final = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_final, # 使用全局
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist'
    )
    model_final.fit(X_all_imputed, y)

    # 4. 保存模型和插补器
    try:
        joblib.dump(model_final, os.path.join(base_path, f'xgb_model_{outcome}.joblib'))
        joblib.dump(imputer_final, os.path.join(base_path, f'simple_imputer_{outcome}.joblib'))
    except Exception as e:
        print(f"Error saving model for {outcome}: {e}")
    # (无 Scaler，故不保存)
    # --------------------------------------------------

    # ------ Bootstrap CI（以 OOF 概率为基准重采样）------
    bootstrap_metrics = {
        'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [],
        'PPV': [], 'NPV': [], 'MCC': [], 'F1 Score': []
    }

    n = len(y)
    for _ in range(BOOTSTRAP_ITERATIONS):
        idx = np.random.choice(n, n, replace=True)
        y_s = y.iloc[idx].reset_index(drop=True)
        p_s = y_pred_prob[idx]
        w_s = weights_eval[idx]

        if len(np.unique(y_s)) < 2:
            continue

        fpr_s, tpr_s, thr_s = roc_curve(y_s, p_s)
        auc_s = auc(fpr_s, tpr_s)
        bootstrap_metrics['AUC'].append(auc_s)

        # 每次重采样内重新用 Youden 选阈值
        opt_idx_s = int(np.argmax(tpr_s - fpr_s))
        thr_opt_s = float(thr_s[opt_idx_s])
        pred_s = (p_s >= thr_opt_s).astype(int)

        try:
            acc_s, sn_s, sp_s, ppv_s, npv_s, mcc_s, f1_s = compute_metrics(y_s, pred_s, w_s)
            bootstrap_metrics['Accuracy'].append(acc_s)
            bootstrap_metrics['Sensitivity'].append(sn_s)
            bootstrap_metrics['Specificity'].append(sp_s)
            bootstrap_metrics['PPV'].append(ppv_s)
            bootstrap_metrics['NPV'].append(npv_s)
            bootstrap_metrics['MCC'].append(mcc_s)
            bootstrap_metrics['F1 Score'].append(f1_s)
        except Exception:
            continue

    # 计算 CI 并格式化
    def fmt_interval(point, lo_hi):
        return f"{format_value(point)} ({format_value(lo_hi[0])} - {format_value(lo_hi[1])})"

    ci = {m: get_ci(v) for m, v in bootstrap_metrics.items()}
    auc_str = fmt_interval(roc_auc, ci['AUC'])
    acc_str = fmt_interval(acc, ci['Accuracy'])
    sn_str  = fmt_interval(sn,  ci['Sensitivity'])
    sp_str  = fmt_interval(sp,  ci['Specificity'])
    ppv_str = fmt_interval(ppv, ci['PPV'])
    npv_str = fmt_interval(npv, ci['NPV'])
    mcc_str = fmt_interval(mcc, ci['MCC'])
    f1_str  = fmt_interval(f1,  ci['F1 Score'])

    # ------- 保存 ROC 图 -------
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'XGB (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {outcome}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(base_path, f'roc_curve_{outcome}.png'))
    plt.close()

    # ------- 保存逐个体预测 -------
    results_df = pd.DataFrame({
        'Participant ID': data['Participant ID'],
        'Predicted_Score': y_pred_prob,
        'Actual_Label': y,
        'Optimal_Threshold': [optimal_threshold] * len(y),
        'Predicted_Label': predictions
    })
    results_df.to_csv(os.path.join(base_path, f'patient_predictions_{outcome}.csv'), index=False)

    return {
        'outcome': outcome,
        'numy': num_positive,
        'AUC': auc_str,
        'Accuracy': acc_str,
        'Sensitivity (Sn)': sn_str,
        'Specificity (Sp)': sp_str,
        'PPV': ppv_str,
        'NPV': npv_str,
        'MCC': mcc_str,
        'F1 Score': f1_str
    }

# ---------------- 并行跑所有结局 ----------------
n_cores = os.cpu_count() or 4
final_results = Parallel(n_jobs=min(max(1, n_cores // 2), len(outcomes)))(
    delayed(process_outcome)(outcome) for outcome in outcomes
)

# ---------------- 汇总保存 ----------------
final_df = pd.DataFrame(final_results)
final_df.to_csv(os.path.join(base_path, 'model_results.csv'), index=False)

# ---------------- 【新增】保存事件计数 ----------------
event_counts_df = final_df[['outcome', 'numy']].copy()
event_counts_df.rename(columns={'outcome': 'Event', 'numy': 'Count'}, inplace=True)
# 按数量降序排列
event_counts_df = event_counts_df.sort_values(by='Count', ascending=False)
event_counts_df.to_csv(os.path.join(base_path, 'event_counts.csv'), index=False)


print("Model results:")
print(final_df)
print("\nEvent counts:")
print(event_counts_df)
print(f"All results saved to {base_path}")