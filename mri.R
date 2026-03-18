suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(broom)
  library(readr)
  library(stringr)
  library(foreach)
  library(doParallel)
  library(parallel)
})





### ====== 配置区（按需修改） ======
#input_file_default <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/mri19/pro5317_morbidity_delphi_mri.csv"   # 默认文件（放在 working directory）
input_file_default <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/mri19/pro5317_morbidity_delphi_mri_no_baseline_cvd.csv"   # 默认文件（放在 working directory）

use_file_choose <- FALSE                      # TRUE 则弹出选择文件对话框
prot_col_start <- 2      # 蛋白列 起始（1-based）
prot_col_end   <- 2921     # 蛋白列 结束（1-based）
n_cmr_last     <- 19     # CMR 列数，取文件最后 n_cmr_last 列
zscore_proteins <- FALSE  # 是否对蛋白列做 z-score（已归一化，不需要）
# 协变量列名
continuous_covs <- c('sample_age_days', 'BMI', 'Age', 'TDI', 'Fasting_time')
discrete_covs <- c('season_binary', 'ethnicity', 'Alcohol_intake_frequency_delphi', 'Current_tobacco_smoking_delphi', 'Sex')
# 诊断列名
diagnosis_col <- 'zhen_need_diagnosis'
# 排除的诊断码例外（I 开头但不排除）
exclude_exceptions <- c('I79', 'I80', 'I81', 'I82', 'I83', 'I84', 'I85', 'I86', 'I87', 'I88', 'I89', 'I95', 'I97', 'I98', 'I99')
# 可选：若你有 outcome -> category 映射表（CSV 两列: outcome, category），可设置路径
outcome_category_map_file <- NULL
# 最小样本量门槛（小于该值将跳过该回归）
min_N <- 30
# 并行核心数：使用70% CPU
num_cores <- 100
### =================================

# 选择文件
input_file <- if (use_file_choose) file.choose() else input_file_default
if (!file.exists(input_file)) {
  stop("找不到输入文件: ", input_file, "。请确认文件名或把 use_file_choose 设为 TRUE 用对话框选择。")
}
message("Reading file: ", input_file)
df <- fread(input_file, data.table = FALSE)

ncol_df <- ncol(df)
if (ncol_df < 10) stop("列数太少，请检查输入 CSV。")

# 基本列定位
id_col <- names(df)[1]

# 蛋白列（以索引为准）
if (prot_col_end > ncol_df) stop("prot_col_end 超出数据列范围，请修改配置。")
prot_cols <- names(df)[prot_col_start:prot_col_end]

# CMR 列：取最后 n_cmr_last 列
if (n_cmr_last > (ncol_df - 1)) stop("n_cmr_last 似乎太大，请检查配置。")
cmr_cols <- tail(names(df), n_cmr_last)

# 检查所有协变量和诊断列是否存在
missing_cols <- setdiff(c(continuous_covs, discrete_covs, diagnosis_col), names(df))
if (length(missing_cols) > 0) {
  stop("缺失列: ", paste(missing_cols, collapse = ", "), "。请确认数据文件包含这些列。")
}

message("ID: ", id_col)
message("Protein cols: ", paste(prot_cols, collapse = ", "))
message("CMR outcome cols: ", paste(cmr_cols, collapse = ", "))
message("Continuous covariates: ", paste(continuous_covs, collapse = ", "))
message("Discrete covariates: ", paste(discrete_covs, collapse = ", "))
message("Diagnosis col: ", diagnosis_col)
message("Using ", num_cores, " cores for parallel processing.")

# 函数：查找站点列
find_site_column <- function(df) {
  candidates <- c('UK Biobank assessment centre | Instance 0')
  for (c in candidates) {
    if (c %in% names(df)) {
      return(c)
    }
  }
  return(NULL)
}

# 函数：缺失值填补
impute_data <- function(df) {
  # 连续型协变量
  for (col in c('sample_age_days', 'Age', 'Fasting_time')) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
  }
  
  # BMI: 按性别分组中位数
  if ('Sex' %in% names(df)) {
    bmi_med_by_sex <- df %>%
      group_by(Sex) %>%
      summarise(med_bmi = median(BMI, na.rm = TRUE)) %>%
      ungroup()
    for (s in unique(df$Sex)) {
      med <- bmi_med_by_sex$med_bmi[bmi_med_by_sex$Sex == s]
      if (length(med) == 0) next
      df$BMI[df$Sex == s & is.na(df$BMI)] <- med
    }
    # 如果还有 NA（例如 Sex 缺失），用整体中位数
    overall_med_bmi <- median(df$BMI, na.rm = TRUE)
    df$BMI[is.na(df$BMI)] <- overall_med_bmi
  }
  
  # TDI: 按站点分组中位数
  site_col <- find_site_column(df)
  if (!is.null(site_col)) {
    tdi_med_by_site <- df %>%
      group_by(.data[[site_col]]) %>%
      summarise(med_tdi = median(TDI, na.rm = TRUE)) %>%
      ungroup()
    for (s in unique(df[[site_col]])) {
      med <- tdi_med_by_site$med_tdi[tdi_med_by_site[[site_col]] == s]
      if (length(med) == 0) next
      df$TDI[df[[site_col]] == s & is.na(df$TDI)] <- med
    }
    # 如果还有 NA，用整体中位数
    overall_med_tdi <- median(df$TDI, na.rm = TRUE)
    df$TDI[is.na(df$TDI)] <- overall_med_tdi
  } else {
    warning("未找到站点列，TDI 使用整体中位数填补。")
    overall_med_tdi <- median(df$TDI, na.rm = TRUE)
    df$TDI[is.na(df$TDI)] <- overall_med_tdi
  }
  
  # 离散型协变量
  for (col in discrete_covs) {
    if (col == 'ethnicity') {
      df[[col]][is.na(df[[col]])] <- 1
    } else {
      tab <- table(df[[col]], useNA = "no")
      mode_val <- as.numeric(names(tab)[which.max(tab)])
      df[[col]][is.na(df[[col]])] <- mode_val
    }
  }
  
  return(df)
}

# 排除基线 CVD 参与者
exclude_cvd <- function(df, diagnosis_col, exclude_exceptions) {
  # 解析诊断码
  df <- df %>%
    mutate(diag_codes = str_split(.data[[diagnosis_col]], fixed("|"))) %>%
    mutate(diag_codes = lapply(diag_codes, str_trim)) %>%
    rowwise() %>%
    mutate(has_cvd = any(
      sapply(diag_codes, function(code) {
        (str_starts(code, "I") && !(code %in% exclude_exceptions)) ||
          str_starts(code, "G45")
      })
    )) %>%
    ungroup() %>%
    filter(!has_cvd) %>%
    select(-diag_codes, -has_cvd)
  
  return(df)
}

# 执行缺失值填补（仅对协变量）
message("Imputing missing values in covariates...")
df <- impute_data(df)

# 执行排除
message("Excluding participants with baseline CVD...")
df <- exclude_cvd(df, diagnosis_col, exclude_exceptions)
if (nrow(df) == 0) stop("排除后无数据剩余。请检查诊断数据或排除规则。")

# 如果 outcome 名里包含 " | Instance 2" 或 " | Instance 3" 等，去除冗余以便输出更整洁
clean_name <- function(x) {
  # 去掉管道及 Instance 信息，收尾空白
  x2 <- str_replace_all(x, "\\s*\\|\\s*Instance\\s*\\d+", "")
  x2 <- str_replace_all(x2, "\\s*\\|\\s*Array\\s*\\d+", "")
  x2 <- str_trim(x2)
  return(x2)
}

orig_cmr_cols <- cmr_cols
cmr_clean <- vapply(orig_cmr_cols, clean_name, FUN.VALUE = character(1), USE.NAMES = FALSE)

# 强制转 numeric（蛋白与 CMR 和连续协变量）
df[prot_cols] <- lapply(df[prot_cols], function(x) as.numeric(as.character(x)))
df[orig_cmr_cols] <- lapply(df[orig_cmr_cols], function(x) as.numeric(as.character(x)))
df[continuous_covs] <- lapply(df[continuous_covs], function(x) as.numeric(as.character(x)))
# 离散协变量转为 factor（但在公式中处理）
df[discrete_covs] <- lapply(df[discrete_covs], as.factor)

# Z-score standardization for proteins and CMR data
message("Performing z-score standardization on proteins and CMR columns...")
for (col in prot_cols) {
  df[[col]] <- as.vector(scale(df[[col]]))
}
for (col in orig_cmr_cols) {
  df[[col]] <- as.vector(scale(df[[col]]))
}

# 读取 outcome->category map（可选）
outcome_map <- NULL
if (!is.null(outcome_category_map_file) && file.exists(outcome_category_map_file)) {
  outcome_map <- read_csv(outcome_category_map_file, col_types = cols())
  # expect columns: outcome, category
}

# 设置并行集群
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# 结果容器
res_list <- list()

res_list <- foreach(i_prot = seq_along(prot_cols), .combine = 'c', .packages = c("dplyr", "broom", "stringr")) %dopar% {
  prot <- prot_cols[i_prot]
  local_res <- list()
  li <- 1L
  for (i_out in seq_along(orig_cmr_cols)) {
    outcome_col <- orig_cmr_cols[i_out]
    outcome_label <- cmr_clean[i_out]  # clean name for output
    
    # subset & drop NA（包括蛋白、outcome 和所有协变量）
    select_cols <- c(outcome_col, prot, continuous_covs, discrete_covs)
    tmp <- df %>% select(all_of(select_cols))
    tmp <- tmp %>% filter(complete.cases(.))
    N <- nrow(tmp)
    if (N < min_N) {
      message(sprintf("跳过: %s ~ %s (N=%d < %d)", prot, outcome_label, N, min_N))
      next
    }
    
    # 构建公式 (连续协变量直接，离散用 as.factor，防止特殊字符用 backtick)
    cont_terms <- paste0("`", continuous_covs, "`", collapse = " + ")
    disc_terms <- paste0("as.factor(`", discrete_covs, "`)", collapse = " + ")
    cov_terms <- paste(cont_terms, disc_terms, sep = " + ")
    fmla_text <- paste0("`", outcome_col, "` ~ `", prot, "` + ", cov_terms)
    fmla <- as.formula(fmla_text)
    
    fit <- tryCatch(lm(fmla, data = tmp), error = function(e) {
      warning("lm 错误: ", conditionMessage(e))
      NULL
    })
    if (is.null(fit)) next
    
    td <- broom::tidy(fit)
    prot_row <- td %>% filter(term == paste0("`", prot, "`"))  # 匹配 backtick 后的 term
    if (nrow(prot_row) == 0) {
      # 若没找到精确 term，尝试模糊匹配
      prot_row <- td %>% filter(str_detect(term, fixed(prot)))
      if (nrow(prot_row) == 0) next
    }
    
    beta <- prot_row$estimate[1]
    se   <- prot_row$std.error[1]
    tval <- prot_row$statistic[1]
    pval <- prot_row$p.value[1]
    
    if (!is.null(outcome_map)) {
      mm <- outcome_map %>% filter(outcome == outcome_label)
      catname <- if (nrow(mm) > 0) mm$category[1] else NA_character_
      local_res[[li]] <- tibble(
        CMR_category = catname,
        Protein = prot,
        Outcome = outcome_label,
        beta = beta,
        SE = se,
        t_value = tval,
        P_value = pval,
        N = N
      )
    } else {
      local_res[[li]] <- tibble(
        Protein = prot,
        Outcome = outcome_label,
        beta = beta,
        SE = se,
        t_value = tval,
        P_value = pval,
        N = N
      )
    }
    li <- li + 1L
  }
  local_res
}

# 停止集群
stopCluster(cl)

if (length(res_list) == 0) stop("未生成任何结果（所有回归均被跳过）。请检查数据与设置。")

res_df <- bind_rows(res_list) %>% arrange(P_value)
res_df <- res_df %>% mutate(p_adj_BH = p.adjust(P_value, method = "BH"))

# 保存结果
out_file <- sub("\\.csv$", "_protein_cmr_assoc_resultsz1.csv", basename(input_file))
readr::write_csv(res_df, out_file)
message("结果已保存为: ", out_file)
fields <- if (!is.null(outcome_map)) "CMR_category, Protein, Outcome, beta, SE, t_value, P_value, N, p_adj_BH" else "Protein, Outcome, beta, SE, t_value, P_value, N, p_adj_BH"
message("结果字段: ", fields)