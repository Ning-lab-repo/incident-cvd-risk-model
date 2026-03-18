
need <- c("data.table","dplyr","stringr","readxl","writexl",
          "TwoSampleMR","ieugwasr","tibble","purrr",
          "parallel") # !! 新增
newp <- need[!need %in% installed.packages()[,1]]
if(length(newp)) install.packages(newp, repos=c("https://mrcieu.r-universe.dev","https://cran.r-project.org"))
invisible(lapply(need, library, character.only=TRUE))


setwd("/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/MR_re_fig_Combind")

disease_dir <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/gwas_fenngen/"

s10_base_dir <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/MR2/Combind_egene_gwas/"

protein_map <- data.frame(
  stringsAsFactors = FALSE,
  name = c("HAVCR1", "CDCP1", "ACTA2", "NTproBNP", "GDF15", "MMP12", "CXCL17", 
           "EDA2R", "BCAN", "WFDC2", "RBFOX3", "HSPB6", "CDHR2"),
  

  folder = c(
    "HAVCR1_Q96D42_OID21422_v1_Oncology",
    "CDCP1_Q9H5V8_OID20940_v1_Neurology",
    "ACTA2_P62736_OID20079_v1_Cardiometabolic",
    "NTproBNP_NTproBNP_OID20125_v1_Cardiometabolic",
    "GDF15_Q99988_OID20251_v1_Cardiometabolic",
    "MMP12_P39900_OID21439_v1_Oncology",
    "CXCL17_Q6UXB2_OID20622_v1_Inflammation",
    "EDA2R_Q9HAV5_OID21451_v1_Oncology",
    "BCAN_Q96GW7_OID20998_v1_Neurology",
    "WFDC2_Q14508_OID21505_v1_Oncology",
    "RBFOX3_A6NFN3_OID30984_v1_Neurology_II",
    "HSPB6_O14558_OID21408_v1_Oncology",
    "CDHR2_Q9BYE9_OID21282_v1_Oncology"
  )
)

# 'path' - 目标 *文件夹* 的完整路径 (for read_s10_data)
protein_map$path <- file.path(s10_base_dir, protein_map$folder)



disease_filenames <- c(
  "finngen_R9_I9_AF.gz",
  "finngen_R9_I9_ANGINA.gz",
  "finngen_R9_I9_ARTEMBTHRNAS.gz",
  "finngen_R9_I9_ATHSCLE.gz",
  "finngen_R9_I9_CARDARR.gz",
  "finngen_R9_I9_CARDMYO.gz",
  "finngen_R9_I9_HYPTENSESS.gz",
  "finngen_R9_I9_HYPTENSRD.gz",
  "finngen_R9_I9_OTHARR.gz",
  "finngen_R9_I9_OTHPER.gz",
  "finngen_R9_I9_PAROXTAC.gz",
  "finngen_R9_I9_PERICAOTH.gz",
  "finngen_R9_I9_PULMEMB.gz",
  "finngen_R9_I9_SECONDRIGHT.gz",
  "finngen_R9_I9_SEQULAE.gz",
  "finngen_R9_I9_TIA.gz"
)

# 构建完整路径
disease_files_paths <- file.path(disease_dir, disease_filenames)

# 从 "finngen_R9_I9_AF.gz" 提取 "AF"
disease_names_match <- stringr::str_match(disease_filenames, "finngen_R9_I9_(.*)\\.gz$")

# 创建 map
disease_map <- data.frame(
  name = disease_names_match[, 2], 
  path = disease_files_paths
)

# 检查所有文件是否存在
missing_files <- disease_map$path[!file.exists(disease_map$path)]
if(length(missing_files) > 0) {
  stop("!!! 严重错误: 找不到以下指定的疾病文件:\n", paste(missing_files, collapse="\n"))
}



out_csv_S13_all <- "S13_ALL_protein_to_disease.csv"
out_csv_S14_all <- "S14_ALL_disease_to_protein.csv"


s13_temp_dir <- "s13_temp_results"
s14_temp_dir <- "s14_temp_results"


## -------- PLINK 和参考面板路径 ----------
plink_path   <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/LD_ref_EUR/plink/plink"
ref_prefix   <- "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/LD_ref_EUR/EUR"

## -------- 小工具 ----------
to_df <- function(x) as.data.frame(x, stringsAsFactors = FALSE)

## -------- 疾病(Finngen) 读取与规范化 ----------
# !! read_finngen_file 无需更改, data.table::fread 可以直接读 .gz
read_finngen_file <- function(fpath){
  if(!file.exists(fpath)) stop("未找到 Finngen 文件：", fpath)
  message(sprintf("  -> 正在读取 Finngen 文件: %s", basename(fpath)))
  dt <- data.table::fread(fpath, sep="\t", header=TRUE, showProgress=FALSE)
  names(dt) <- sub("^#","",names(dt))
  req <- c("chrom","pos","ref","alt","rsids","pval","beta")
  if(!all(req %in% names(dt))) stop("Finngen 文件缺少：", paste(setdiff(req, names(dt)), collapse=", "))
  if(!"sebeta" %in% names(dt)){
    if("se" %in% names(dt)) names(dt)[names(dt)=="se"] <- "sebeta"
    else stop("Finngen 文件缺少标准误列：sebeta/se")
  }
  if(!"af_alt" %in% names(dt)){
    cand <- intersect(c("af","eaf","eaf_alt"), names(dt))
    if(length(cand)) names(dt)[names(dt)==cand[1]] <- "af_alt" else {
      dt$af_alt <- NA_real_
    }
  }
  to_df(dt)
}

# !! (原 format_outcome_AF) -> 已参数化
format_outcome_disease <- function(disease_dt, disease_name){
  out <- disease_dt %>%
    dplyr::rename(SNP = rsids, effect_allele = alt, other_allele = ref,
                  eaf = af_alt, chr = chrom, pos = pos) %>%
    dplyr::mutate(beta = as.numeric(beta), se = as.numeric(sebeta), pval = as.numeric(pval),
                  chr = as.character(chr), pos = as.integer(pos)) %>%
    dplyr::filter(!is.na(SNP),
                  !is.na(chr), !is.na(pos),
                  effect_allele %in% c("A","C","G","T"),
                  other_allele  %in% c("A","C","G","T")) %>%
    to_df()
  out$id.outcome <- disease_name
  out$outcome    <- disease_name
  message(sprintf("  -> 疾病 %s 格式化完成，共 %d 行有效变异。", disease_name, nrow(out)))
  out
}

## -------- S10 (蛋白) 读取与规范化 ----------
# (代码与原来一致，保持不变)
read_s10_data <- function(dir_path) {
  files <- list.files(dir_path, pattern = "\\.gz$", full.names = TRUE)
  if(length(files) == 0) {
    message(sprintf("  -> S10 警告: 在 %s 中未找到 .gz 文件", dir_path))
    return(NULL) # 返回 NULL 以便跳过
  }
  
  message(sprintf("  -> 正在从 %s 读取 %d 个 S10 .gz 文件...", basename(dir_path), length(files)))
  
  all_data <- purrr::map_dfr(files, ~ data.table::fread(.x, header = TRUE, showProgress = FALSE)) %>% 
    to_df()
  
  nm <- names(all_data)
  need <- c("ID", "CHROM", "GENPOS", "ALLELE0", "ALLELE1", "A1FREQ", "BETA", "SE", "LOG10P")
  if(!all(need %in% nm)) stop("S10 文件缺少：", paste(setdiff(need, nm), collapse=", "))
  
  out <- all_data %>%
    dplyr::transmute(
      chr = as.character(CHROM),
      pos = as.integer(GENPOS),
      SNP_ID_raw = ID,
      effect_allele = ALLELE1,
      other_allele  = ALLELE0,
      eaf  = as.numeric(A1FREQ),
      beta = as.numeric(BETA),
      se   = as.numeric(SE),
      pval = 10^(-as.numeric(LOG10P))
    ) %>%
    dplyr::filter(!is.na(pval),
                  !is.na(chr), !is.na(pos),
                  effect_allele %in% c("A","C","G","T"),
                  other_allele  %in% c("A","C","G","T")) %>%
    to_df()
  
  message(sprintf("  -> S10 数据读取完成，共 %d 行有效变异。", nrow(out)))
  return(out)
}

## -------- 分批安全 clumping（本地 PLINK） ----------
# (代码与原来一致，保持不变)
safe_ld_clump <- function(dat, clump_r2=0.01, clump_kb=1000,
                          batch_size=1000, max_tries=3, base_sleep=0.6){
  dat <- dat %>% dplyr::distinct(rsid, .keep_all = TRUE) %>% to_df()
  n <- nrow(dat)
  if(n == 0) return(dat[0, , drop=FALSE])
  idx <- split(seq_len(n), ceiling(seq_len(n)/batch_size))
  out <- list()
  for(i in seq_along(idx)){
    sub <- dat[idx[[i]], , drop=FALSE]
    attempt <- 1
    repeat{
      # !! 关键：res 是 try(...) 的结果
      res <- try(
        local_ld_clump(dat = to_df(sub), clump_r2 = clump_r2, clump_kb = clump_kb),
        silent = TRUE
      )
      # !! 关键：检查 res 是否 *不是* "try-error"
      if(!inherits(res, "try-error")){
        out[[length(out)+1]] <- to_df(res)
        break
      } else {
        # !! 关键：如果 local_ld_clump (已修复) stop() 了, 会进入这里
        if(attempt >= max_tries){
          stop("本地 ld_clump 在第 ", i, " 批失败 (3次尝试后)：", as.character(res))
        }
        message(sprintf("  -> Clumping 批次 %d 失败 (尝试 %d/%d)，%f 秒后重试...", i, attempt, max_tries, base_sleep * attempt))
        Sys.sleep(base_sleep * attempt)
        attempt <- attempt + 1
      }
    }
    Sys.sleep(0.4)
  }
  dplyr::bind_rows(out) %>% to_df()
}


local_ld_clump <- function(dat, clump_r2=0.01, clump_kb=1000){
  temp_pval <- tempfile(fileext = ".pval")
  temp_out  <- tempfile()
  on.exit(unlink(c(temp_pval, paste0(temp_out, c(".clumped", ".log", ".noscript")))), add = TRUE)
  
  # 增加安全检查
  if(nrow(dat) == 0) {
    return(dat[0, , drop=FALSE]) 
  }
  if(!"rsid" %in% names(dat) || !"pval" %in% names(dat)) {
    stop("local_ld_clump: 'dat' 缺少 'rsid' 或 'pval' 列。")
  }
  
  write.table(dat[, c("rsid", "pval")], file = temp_pval, col.names = c("SNP", "P"), 
              row.names = FALSE, quote = FALSE)
  
  cmd <- paste(plink_path, "--bfile", ref_prefix, "--clump", temp_pval, 
               "--clump-p1 1 --clump-r2", clump_r2, "--clump-kb", clump_kb, 
               "--out", temp_out)
  system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
  
  clumped_file <- paste0(temp_out, ".clumped")
  
  # !! (已修复) 恢复了 stop() 逻辑
  if(!file.exists(clumped_file) || file.info(clumped_file)$size == 0){
    log_file <- paste0(temp_out, ".log")
    log_content <- " (log 文件不存在或为空)"
    if(file.exists(log_file)) {
      try({
        log_content <- paste(readLines(log_file, n=20), collapse="\n")
      }, silent=TRUE)
    }
    # !! (已修复) 恢复了 stop()
    stop("PLINK clumping 失败，未生成 .clumped 文件。\nPLINK 日志: \n", log_content)
  }
  
  clumped <- data.table::fread(clumped_file, header = TRUE, showProgress = FALSE) %>% to_df()
  keep <- unique(clumped$SNP)
  
  dplyr::filter(dat, rsid %in% keep) %>% to_df()
}

## -------- RSID 映射表创建 (新函数) ----------
create_rsid_lookup <- function(disease_data_full) {
  message(sprintf("  -> 正在为 %s 创建 RSID 映射表...", disease_data_full$outcome[1]))
  rsid_lookup <- disease_data_full %>%
    dplyr::select(chr, pos, effect_allele, other_allele, rsid = SNP) %>%
    dplyr::filter(!is.na(rsid)) %>%
    dplyr::distinct(chr, pos, effect_allele, other_allele, .keep_all = TRUE)
  
  rsid_lookup_rev <- disease_data_full %>%
    dplyr::select(chr, pos, effect_allele_rev = other_allele, other_allele_rev = effect_allele, rsid = SNP) %>%
    dplyr::filter(!is.na(rsid)) %>%
    dplyr::rename(effect_allele = effect_allele_rev, other_allele = other_allele_rev) %>%
    dplyr::distinct(chr, pos, effect_allele, other_allele, .keep_all = TRUE)
  
  rsid_lookup_combined <- dplyr::bind_rows(rsid_lookup, rsid_lookup_rev) %>%
    dplyr::distinct(chr, pos, effect_allele, other_allele, .keep_all = TRUE)
  
  message(sprintf("  -> RSID 映射表创建完毕，共 %d 个独特 SNP 映射。", nrow(rsid_lookup_combined)))
  return(rsid_lookup_combined)
}


## -------- S10: 构建蛋白暴露 (!! 已修改：依赖 rsid_lookup) ----------
build_exposure_from_s10 <- function(full_s10_dat, protein_symbol, rsid_lookup){
  
  exp0 <- full_s10_dat 
  
  # --- 内部辅助函数：尝试一个阈值 ---
  try_threshold <- function(p_thresh, thr_label) {
    message(sprintf("  -> [%s] (暴露) 尝试阈值 %s...", protein_symbol, thr_label))
    
    exp1 <- dplyr::filter(exp0, pval < p_thresh) %>% to_df()
    
    exp1_with_rsid <- exp1 %>%
      # !! 关键：使用传入的、特定于疾病的 rsid_lookup
      dplyr::left_join(rsid_lookup, by = c("chr", "pos", "effect_allele", "other_allele")) %>%
      dplyr::filter(!is.na(rsid))
    
    message(sprintf("  -> [%s] (%s): %d 原始 SNPs -> %d 映射 RSIDs.", 
                    protein_symbol, thr_label, nrow(exp1), nrow(exp1_with_rsid)))
    
    if(nrow(exp1_with_rsid) < 3) {
      message(sprintf("  -> [%s] (%s): 映射后 IVs < 3 (n=%d)，此阈值失败。", 
                      protein_symbol, thr_label, nrow(exp1_with_rsid)))
      return(NULL)
    }
    
    message(sprintf("  -> [%s] (%s): 找到 %d 个已映射的 SNPs，准备 clumping...",
                    protein_symbol, thr_label, nrow(exp1_with_rsid)))
    
    clumped_res <- try(
      safe_ld_clump(
        dat = dplyr::tibble(rsid = exp1_with_rsid$rsid, pval = exp1_with_rsid$pval, id = protein_symbol),
        clump_r2 = 0.01, clump_kb = 1000
      ),
      silent = TRUE
    )
    
    if (inherits(clumped_res, "try-error")) {
      message(sprintf("  -> [%s] (%s): Clumping 步骤失败: %s", 
                      protein_symbol, thr_label, as.character(clumped_res)))
      return(NULL)
    }
    
    keep_rsids <- unique(clumped_res$rsid)
    exp2 <- exp1_with_rsid %>%
      dplyr::filter(rsid %in% keep_rsids) %>%
      dplyr::mutate(SNP = rsid)
    
    message(sprintf("  -> [%s] (%s): Clumping 完成，保留 %d 个 IVs。", 
                    protein_symbol, thr_label, nrow(exp2)))
    
    if(nrow(exp2) < 3) {
      message(sprintf("  -> [%s] (%s): Clumping 后 IVs < 3 (n=%d)，此阈值失败。", 
                      protein_symbol, thr_label, nrow(exp2)))
      return(NULL)
    }
    
    exp2$id.exposure <- protein_symbol
    exp2$exposure    <- protein_symbol
    exp2$.__thr__    <- thr_label
    return(to_df(exp2))
  }
  
  # --- 逻辑流 ---
  result_5e8 <- try_threshold(p_thresh = 5e-8, thr_label = "5.00E-08")
  if (!is.null(result_5e8)) {
    return(result_5e8)
  }
  
  message(sprintf("  -> [%s] 5e-8 阈值失败 (最终 IVs < 3)，放宽到 1e-6...", protein_symbol))
  result_1e6 <- try_threshold(p_thresh = 1e-6, thr_label = "1.00E-06")
  
  return(result_1e6)
}


## -------- 疾病暴露 (原 build_AF_exposure) ----------
build_disease_exposure <- function(disease_data_full){
  disease_name <- disease_data_full$outcome[1]
  message(sprintf("  -> 正在构建 %s 的 (暴露) IVs...", disease_name))
  
  exp0 <- disease_data_full %>%
    dplyr::select(SNP,beta,se,eaf,effect_allele,other_allele,pval,chr,pos) %>% to_df()
  
  # --- 内部辅助函数：尝试一个阈值 ---
  try_threshold_disease <- function(p_thresh, thr_label) {
    message(sprintf("  -> [%s] (暴露) 尝试阈值 %s...", disease_name, thr_label))
    
    exp1 <- dplyr::filter(exp0, pval < p_thresh) %>% to_df()
    
    if(nrow(exp1) < 3) {
      message(sprintf("  -> [%s] (%s): 原始 IVs < 3 (n=%d)，此阈值失败。", 
                      disease_name, thr_label, nrow(exp1)))
      return(NULL)
    }
    
    message(sprintf("  -> [%s] (%s): G%d 个 SNP，准备 clumping...", disease_name, thr_label, nrow(exp1)))
    
    clumped_res <- try(
      safe_ld_clump(
        dat = dplyr::tibble(rsid = exp1$SNP, pval = exp1$pval, id = disease_name),
        clump_r2 = 0.01, clump_kb = 1000,
        batch_size = 1000
      ),
      silent = TRUE
    )
    
    if (inherits(clumped_res, "try-error")) {
      message(sprintf("  -> [%s] (%s): Clumping 步骤失败: %s", 
                      disease_name, thr_label, as.character(clumped_res)))
      return(NULL)
    }
    
    keep <- unique(clumped_res$rsid)
    exp2 <- dplyr::filter(exp1, SNP %in% keep) %>% to_df()
    
    message(sprintf("  -> [%s] (%s): Clumping 完成，保留 %d 个 IVs。", disease_name, thr_label, nrow(exp2)))
    
    if(nrow(exp2) < 3) {
      message(sprintf("  -> [%s] (%s): Clumping 后 IVs < 3 (n=%d)，此阈值失败。", 
                      disease_name, thr_label, nrow(exp2)))
      return(NULL)
    }
    
    exp2$id.exposure <- disease_name
    exp2$exposure    <- disease_name
    exp2$.__thr__    <- thr_label
    return(to_df(exp2))
  }
  
  # --- 逻辑流 ---
  result_5e8 <- try_threshold_disease(p_thresh = 5e-8, thr_label = "5.00E-08")
  if (!is.null(result_5e8)) {
    return(result_5e8)
  }
  message(sprintf("  -> [%s] 5e-8 阈值失败 (最终 IVs < 3)，放宽到 1e-6...", disease_name))
  result_1e6 <- try_threshold_disease(p_thresh = 1e-6, thr_label = "1.00E-06")
  return(result_1e6)
}

## -------- 蛋白结局 (!! 已修改：依赖 rsid_lookup) ----------
build_protein_outcome_from_s10 <- function(full_s10_dat, protein_symbol, rsid_lookup){
  
  out <- full_s10_dat %>%
    # !! 关键：使用传入的、特定于疾病的 rsid_lookup
    dplyr::left_join(rsid_lookup, by = c("chr", "pos", "effect_allele", "other_allele")) %>%
    dplyr::filter(!is.na(rsid)) %>% 
    dplyr::mutate(SNP = rsid) 
  
  if(nrow(out) == 0) {
    message(sprintf("  -> [%s] (结局): 没有 S10 SNP 能匹配到 (疾病) rsid。", protein_symbol))
    return(NULL)
  }
  
  out$id.outcome <- protein_symbol
  out$outcome    <- protein_symbol
  message(sprintf("  -> [%s] (结局): 成功构建，%d 个 SNPs 匹配到 rsid。", protein_symbol, nrow(out)))
  to_df(out)
}

## -------- 路径 1：蛋白 -> 疾病 (原 run_mr_protein_to_AF) ----------
run_mr_protein_to_disease <- function(protein_symbol, s10_data, disease_data_full, rsid_lookup){
  
  disease_name <- disease_data_full$outcome[1]
  
  # !! 关键：传入 rsid_lookup
  exp_dat <- build_exposure_from_s10(s10_data, protein_symbol, rsid_lookup)
  
  if(is.null(exp_dat) || nrow(exp_dat) < 3) {
    message(sprintf("  -> [%s -> %s] IVs 不足 (<3)，跳过 S13。", protein_symbol, disease_name))
    return(NULL)
  }
  
  # 使用传入的、已格式化的疾病数据作为结局
  out_dat <- disease_data_full %>% dplyr::filter(SNP %in% exp_dat$SNP) %>% to_df()
  
  if(nrow(out_dat) == 0) {
    message(sprintf("  -> [%s -> %s] 暴露和结局 0 SNP 重叠 (基于 rsid)，跳过 S13。", protein_symbol, disease_name))
    return(NULL)
  }
  
  dat_h <- TwoSampleMR::harmonise_data(
    exposure_dat = TwoSampleMR::format_data(to_df(exp_dat), type="exposure", snp_col="SNP"),
    outcome_dat  = TwoSampleMR::format_data(to_df(out_dat),  type="outcome", snp_col="SNP"),
    action = 2
  )
  if(nrow(dat_h) == 0 || nrow(dat_h %>% dplyr::filter(mr_keep)) < 3) {
    message(sprintf("  -> [%s -> %s] Harmonise 后 SNP 不足 (<3)，跳过 S13。", protein_symbol, disease_name))
    return(NULL)
  }
  
  mr_res <- TwoSampleMR::mr(dat_h, method_list=c(
    "mr_ivw","mr_egger_regression","mr_weighted_median","mr_simple_mode","mr_weighted_mode"
  ))
  Nsnp <- dat_h %>% dplyr::filter(mr_keep) %>% dplyr::distinct(SNP) %>% nrow()
  thr_used <- unique(exp_dat$.__thr__)[1]
  
  mr_res %>%
    dplyr::mutate(
      OR = exp(b),
      CI_lower = exp(b - 1.96*se),
      CI_upper = exp(b + 1.96*se),
      `OR (95% CI)` = sprintf("%.2f (%.2f, %.2f)", OR, CI_lower, CI_upper),
      Threshold = thr_used,
      Outcome = disease_name, # !! 动态结局名称
      Exposure = protein_symbol,
      Nsnp = as.integer(Nsnp)
    ) %>%
    dplyr::select(Outcome, Exposure, Method = method, Nsnp,
                  Beta = b, SE = se, `P value` = pval, `OR (95% CI)`, Threshold) %>%
    to_df()
}

## -------- 路径 2：疾病 -> 蛋白 (原 run_mr_AF_to_protein) ----------
run_mr_disease_to_protein <- function(protein_symbol, s10_data, disease_exp_dat, rsid_lookup){
  
  if(is.null(disease_exp_dat) || nrow(disease_exp_dat) < 3) {

    message(sprintf("  -> [? -> %s] 疾病 IVs 不足 (<3)，跳过 S14。", protein_symbol))
    return(NULL)
  }
  
  disease_name <- disease_exp_dat$exposure[1]
  
  # !! 关键：传入 rsid_lookup
  prot_out <- build_protein_outcome_from_s10(s10_data, protein_symbol, rsid_lookup)
  
  if(is.null(prot_out) || nrow(prot_out) == 0) {
    message(sprintf("  -> [%s -> %s] 蛋白结局数据为空 (无 rsid 匹配)，跳过 S14。", disease_name, protein_symbol))
    return(NULL)
  }
  
  snps <- intersect(disease_exp_dat$SNP, prot_out$SNP)
  if(length(snps) < 3){ 
    message(sprintf("  -> [%s -> %s] 重叠 SNP < 3 (基于 rsid)，跳过 S14。", disease_name, protein_symbol))
    return(NULL)
  }
  
  exp_dat <- disease_exp_dat %>% dplyr::filter(SNP %in% snps) %>% to_df()
  out_dat <- prot_out        %>% dplyr::filter(SNP %in% snps) %>% to_df()
  
  dat_h <- TwoSampleMR::harmonise_data(
    exposure_dat = TwoSampleMR::format_data(to_df(exp_dat), type="exposure", snp_col="SNP"),
    outcome_dat  = TwoSampleMR::format_data(to_df(out_dat),  type="outcome", snp_col="SNP"),
    action = 2
  )
  if(nrow(dat_h) == 0 || nrow(dat_h %>% dplyr::filter(mr_keep)) < 3) {
    message(sprintf("  -> [%s -> %s] Harmonise 后 SNP 不足 (<3)，跳过 S14。", disease_name, protein_symbol))
    return(NULL)
  }
  
  mr_res <- TwoSampleMR::mr(dat_h, method_list=c(
    "mr_ivw","mr_egger_regression","mr_weighted_median","mr_simple_mode","mr_weighted_mode"
  ))
  Nsnp <- dat_h %>% dplyr::filter(mr_keep) %>% dplyr::distinct(SNP) %>% nrow()
  thr_used <- unique(disease_exp_dat$.__thr__)[1]
  
  mr_res %>%
    dplyr::mutate(
      OR = exp(b),
      CI_lower = exp(b - 1.96*se),
      CI_upper = exp(b + 1.96*se),
      `OR (95% CI)` = sprintf("%.2f (%.2f, %.2f)", OR, CI_lower, CI_upper),
      Threshold = thr_used,
      Outcome = protein_symbol,
      Exposure = disease_name, # !! 动态暴露名称
      Nsnp = as.integer(Nsnp)
    ) %>%
    dplyr::select(Outcome, Exposure, Method = method, Nsnp,
                  Beta = b, SE = se, `P value` = pval, `OR (95% CI)`, Threshold) %>%
    to_df()
}




# !! (新) 步骤 0: 创建临时目录
message("--- 正在创建临时输出目录 ---")
dir.create(s13_temp_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(s14_temp_dir, showWarnings = FALSE, recursive = TRUE)


message(sprintf("\n--- 开始 MR 分析循环： %d 种疾病 vs %d 种蛋白 ---", nrow(disease_map), nrow(protein_map)))

# 1. !! 定义并行工作函数 (Worker Function) !!
# 这个函数包含了原来 "for (i in ...)" 循环的 *所有* 内容
run_analysis_for_one_disease <- function(i) {
  
  current_disease_name <- disease_map$name[i]
  current_disease_path <- disease_map$path[i]
  
  message(sprintf("\n--- [!! 并行任务启动 !!] 开始处理: %s ---", current_disease_name))
  
  # --- 步骤 D1: 加载和格式化当前疾病数据 (作为结局)
  disease_dt <- read_finngen_file(current_disease_path)
  disease_data_full <- format_outcome_disease(disease_dt, current_disease_name)
  
  # --- 步骤 D2: 为当前疾病创建 RSID 映射表
  current_rsid_lookup <- create_rsid_lookup(disease_data_full)
  
  # --- 步骤 D3: 构建当前疾病的 IVs (作为暴露)
  # !! 关键: 这里的 try() 是为了捕捉 local_ld_clump 抛出的 stop()
  current_disease_exposure_dat <- try(
    build_disease_exposure(disease_data_full),
    silent = TRUE
  )
  
  if (inherits(current_disease_exposure_dat, "try-error")) {
    message(sprintf("  -> !!! 严重错误: 无法为 %s 构建暴露 IVs (Clumping失败)。将跳过所有 S14 分析。\n  -> 错误: %s", 
                    current_disease_name, as.character(current_disease_exposure_dat)))
    current_disease_exposure_dat <- NULL # 确保其为 NULL
  } else if (is.null(current_disease_exposure_dat)) {
    message(sprintf("  -> !!! 警告: 无法为 %s 构建暴露 IVs (IVs < 3)。将跳过所有 S14 (疾病 -> 蛋白) 分析。", current_disease_name))
  }
  
  # 1b. !! (新) 定义 *内层* 工作函数
  #     这个函数将在 *串行* for 循环中被调用
  run_one_protein_pair <- function(j) {
    
    current_protein_name <- protein_map$name[j]
    current_protein_path <- protein_map$path[j]
    
    message(sprintf("\n  --- [%s - 蛋白 %d/%d] %s vs %s ---", current_disease_name, j, nrow(protein_map), current_protein_name, current_disease_name))
    
    # --- 步骤 P1: 检查蛋白文件夹是否存在
    if (!dir.exists(current_protein_path)) {
      message(sprintf("  -> !!! 警告: 找不到蛋白文件夹: %s。跳过 %s。", 
                      current_protein_path, current_protein_name))
      return(TRUE) # 跳到下一个蛋白
    }
    
    # --- 步骤 P2: 加载当前蛋白数据
    s10_full_data <- read_s10_data(current_protein_path)
    if (is.null(s10_full_data)) {
      message(sprintf("  -> !!! 警告: 无法从 %s 读取 S10 数据。跳过 %s。", 
                      current_protein_path, current_protein_name))
      return(TRUE) # 跳到下一个蛋白
    }
    
    # --- 步骤 P3: 运行 路径 1 (蛋白 -> 疾病)
    res_S13 <- try(run_mr_protein_to_disease(
      protein_symbol = current_protein_name,
      s10_data = s10_full_data,
      disease_data_full = disease_data_full,
      rsid_lookup = current_rsid_lookup
    ), silent = TRUE)
    
    if (!inherits(res_S13, "try-error") && !is.null(res_S13) && nrow(res_S13) > 0) {
      # !! (新) 写入单独的 CSV 文件 !!
      out_s13_file <- file.path(s13_temp_dir, sprintf("s13_%s_vs_%s.csv", current_protein_name, current_disease_name))
      data.table::fwrite(res_S13, out_s13_file)
      message(sprintf("  -> S13 (%s -> %s) 结果已保存到临时文件。", current_protein_name, current_disease_name))
    } else if (inherits(res_S13, "try-error")) {
      message(sprintf("  -> S13 (%s -> %s) 运行失败: %s", current_protein_name, current_disease_name, as.character(res_S13)))
    }
    
    # --- 步骤 P4: 运行 路径 2 (疾病 -> 蛋白)
    if (!is.null(current_disease_exposure_dat)) {
      res_S14 <- try(run_mr_disease_to_protein(
        protein_symbol = current_protein_name,
        s10_data = s10_full_data,
        disease_exp_dat = current_disease_exposure_dat,
        rsid_lookup = current_rsid_lookup
      ), silent = TRUE)
      
      if (!inherits(res_S14, "try-error") && !is.null(res_S14) && nrow(res_S14) > 0) {
        # !! (新) 写入单独的 CSV 文件 !!
        out_s14_file <- file.path(s14_temp_dir, sprintf("s14_%s_vs_%s.csv", current_disease_name, current_protein_name))
        data.table::fwrite(res_S14, out_s14_file)
        message(sprintf("  -> S14 (%s -> %s) 结果已保存到临时文件。", current_disease_name, current_protein_name))
      } else if (inherits(res_S14, "try-error")) {
        message(sprintf("  -> S14 (%s -> %s) 运行失败: %s", current_disease_name, current_protein_name, as.character(res_S14)))
      }
      
    } else {
      message(sprintf("  -> 跳过 S14 (%s -> %s)，因为 %s 没有 IVs。", current_disease_name, current_protein_name, current_disease_name))
    }
    
    return(TRUE) # 表示此蛋白对已处理完毕
  } # !! 结束 *内层* 工作函数定义
  
  
  # 3. !! (新) 执行 *内层* 串行循环 (遍历 13 个蛋白)
  message(sprintf("  -> [%s] 启动 *串行* 循环处理 13 个蛋白...", current_disease_name))
  
  # !! (新) 改为普通的 for 循环
  for (j in 1:nrow(protein_map)) {
    tryCatch({
      run_one_protein_pair(j)
    }, error = function(e) {
      message(sprintf("  -> !!! 严重错误 (内层循环 %d): %s vs %s。错误: %s", 
                      j, protein_map$name[j], current_disease_name, as.character(e)))
    })
  }
  
  message(sprintf("--- [!! 并行任务完成 !!] %s ---", current_disease_name))
  
  return(TRUE) # 表示此疾病已处理完毕
  
} # !! 结束 *外层* 并行工作函数 !!


# 2. !! 设置核心数并执行 *外层* 并行 !!
# (总核心数) = (外层核心数)

# 外层核心数：每个疾病一个 (最多 16)
num_outer_cores <- min(nrow(disease_map), parallel::detectCores() - 1, 125)

message(sprintf("--- 准备启动 *非嵌套* 并行 ---"))
message(sprintf("--- 外层 (疾病) 核心: %d", num_outer_cores))
message(sprintf("--- 内层 (蛋白) 循环将 *串行* 运行在每个核心内部。"))
message(sprintf("--- 预计总核心使用: %d ---", num_outer_cores))


# !! mclapply 会将 1, 2, ..., 16 分配给 'run_analysis_for_one_disease' 函数 !!
all_results_list <- parallel::mclapply(
  1:nrow(disease_map), 
  run_analysis_for_one_disease, 
  # !! (新) 移除了 inner_cores 参数
  mc.cores = num_outer_cores
)

message("\n--- 所有并行任务已完成。正在从临时文件合并最终结果... ---")


## -------- 
## --------         合并与保存最终结果         --------
## -------- 

# 3. !! (新) 从 *临时文件* 提取和合并结果 !!

# --- 合并 S13 ---
s13_files <- list.files(s13_temp_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(s13_files) > 0) {
  message(sprintf("正在合并 %d 个 S13 临时文件...", length(s13_files)))
  
  # 使用 data.table 的 rbindlist (高效)
  final_S13_list <- purrr::map(s13_files, data.table::fread)
  final_S13 <- data.table::rbindlist(final_S13_list)
  
  message(sprintf("S13 (蛋白 -> 疾病) 共 %d 行结果。", nrow(final_S13)))
  
  # 计算 FDR
  final_S13$FDR_BH <- p.adjust(final_S13$`P value`, method="BH")
  
  # 保存
  data.table::fwrite(final_S13, out_csv_S13_all)
  message("S13 最终结果已保存到: ", out_csv_S13_all)
  
  # 预览
  message("\n--- 预览 最终 S13 (蛋白 -> 疾病) ---")
  print(tibble::as_tibble(final_S13), n = 20)
  
} else {
  message("S13 (蛋白 -> 疾病) 无有效结果。")
}

# --- 合并 S14 ---
s14_files <- list.files(s14_temp_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(s14_files) > 0) {
  message(sprintf("正在合并 %d 个 S14 临时文件...", length(s14_files)))
  
  # 使用 data.table 的 rbindlist (高效)
  final_S14_list <- purrr::map(s14_files, data.table::fread)
  final_S14 <- data.table::rbindlist(final_S14_list)
  
  message(sprintf("S14 (疾病 -> 蛋白) 共 %d 行结果。", nrow(final_S14)))
  
  # 计算 FDR
  final_S14$FDR_BH <- p.adjust(final_S14$`P value`, method="BH")
  
  # 保存
  data.table::fwrite(final_S14, out_csv_S14_all)
  message("S14 最终结果已保存到: ", out_csv_S14_all)
  
  # 预览
  message("\n--- 预览 最终 S14 (疾病 -> 蛋白) ---")
  print(tibble::as_tibble(final_S14), n = 20)
  
} else {
  message("S14 (疾病 -> 蛋白) 无有效结果。")
}

message("\n--- 脚本运行完毕 ---")
message(sprintf("--- (可选) 临时文件已保留在 %s/ 和 %s/ 中。", s13_temp_dir, s14_temp_dir))
message(sprintf("--- (可选) 可以稍后手动删: unlink('%s', recursive = TRUE); unlink('%s', recursive = TRUE) ---", s13_temp_dir, s14_temp_dir))
