#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

NONLEGEND_TEXT_MULT <- 1.5

scale_nonlegend_text <- function(x, mult = NONLEGEND_TEXT_MULT) {
  as.numeric(x) * mult
}

fmt_p <- function(p) {
  p <- suppressWarnings(as.numeric(p))
  ifelse(!is.finite(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
}

sig_star <- function(p, cutoff = 0.01) {
  p <- suppressWarnings(as.numeric(p))
  ifelse(is.finite(p) & p < cutoff, "*", "")
}

resolve_plot_family <- function(preferred = "Arial") {
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    hit <- tryCatch(systemfonts::match_font(preferred), error = function(e) NULL)
    if (!is.null(hit) && !is.null(hit$path) && nzchar(hit$path) && file.exists(hit$path)) return(preferred)
  }
  "sans"
}

first_existing_path <- function(dir_path, candidates) {
  for (nm in candidates) {
    p <- file.path(dir_path, nm)
    if (file.exists(p)) return(p)
  }
  NA_character_
}

load_summary_csv <- function(res_dir) {
  exact_csv <- first_existing_path(
    res_dir,
    c("landmark4_results_multi_proteins.csv", "landmark_results_multi_proteins.csv")
  )
  if (!is.na(exact_csv) && nzchar(exact_csv)) {
    df <- read.csv(exact_csv, check.names = FALSE, stringsAsFactors = FALSE)
    return(list(df = df, sources = exact_csv))
  }

  summary_csvs <- list.files(
    res_dir,
    pattern = "landmark_results_multi_proteins\\.csv$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  summary_csvs <- summary_csvs[file.exists(summary_csvs)]
  if (length(summary_csvs) == 0) {
    stop("Cannot find summary csv in: ", res_dir)
  }

  # Prefer larger prefix files first (e.g., 8... then 5...) for stable protein ordering.
  bname <- basename(summary_csvs)
  prefix_num <- suppressWarnings(as.numeric(sub("^([0-9]+).*$", "\\1", bname)))
  ord <- order(is.na(prefix_num), -prefix_num, bname)
  summary_csvs <- summary_csvs[ord]

  parts <- lapply(summary_csvs, function(p) read.csv(p, check.names = FALSE, stringsAsFactors = FALSE))
  all_cols <- unique(unlist(lapply(parts, names)))
  parts <- lapply(parts, function(d) {
    miss <- setdiff(all_cols, names(d))
    if (length(miss) > 0) {
      for (m in miss) d[[m]] <- NA
    }
    d[, all_cols, drop = FALSE]
  })
  df <- bind_rows(parts) %>% distinct()
  list(df = df, sources = summary_csvs)
}

parse_bool <- function(x) {
  tolower(trimws(as.character(x))) %in% c("1", "true", "t", "yes", "y")
}

pick_first_nonempty <- function(x) {
  x <- trimws(as.character(x))
  x <- x[!is.na(x) & tolower(x) != "na"]
  x <- x[nzchar(x)]
  if (length(x) == 0) NA_character_ else x[1]
}

get_script_dir <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa) > 0) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/", mustWork = FALSE)))
  normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

is_absolute_path <- function(path) {
  grepl("^[A-Za-z]:[/\\\\]", path) || grepl("^[/\\\\]{2}", path) || grepl("^/", path)
}

resolve_path <- function(path, base_dir) {
  path <- trimws(as.character(path))
  if (!nzchar(path)) return(normalizePath(base_dir, winslash = "/", mustWork = FALSE))
  if (is_absolute_path(path)) normalizePath(path, winslash = "/", mustWork = FALSE) else normalizePath(file.path(base_dir, path), winslash = "/", mustWork = FALSE)
}

parse_args <- function() {
  out <- list(
    results_dir = "../results8",
    figures_dir = ".",
    figure_suffix = "_enlarged",
    no_forest = FALSE,
    figure54_names = TRUE
  )
  args <- commandArgs(trailingOnly = TRUE)
  i <- 1
  while (i <= length(args)) {
    k <- args[i]
    v <- if (i < length(args)) args[i + 1] else ""
    if (k == "--results-dir") out$results_dir <- v
    else if (k == "--figures-dir") out$figures_dir <- v
    else if (k == "--figure-suffix") out$figure_suffix <- v
    else if (k == "--no-forest") out$no_forest <- parse_bool(v)
    else if (k == "--figure54-names") out$figure54_names <- parse_bool(v)
    else if (k %in% c("-h", "--help")) {
      cat(
        "Usage:\n",
        "Rscript redraw_figures8_from_results_csv.R [--results-dir ../results8] [--figures-dir .] [--figure-suffix _enlarged] [--no-forest 1] [--figure54-names 1]\n",
        sep = ""
      )
      quit(save = "no", status = 0)
    } else stop("Unknown argument: ", k)
    i <- i + 2
  }
  out
}

infer_landmark_levels <- function(summary_df) {
  lab <- if ("landmark_label" %in% names(summary_df)) trimws(as.character(summary_df$landmark_label)) else rep("", nrow(summary_df))
  yr <- if ("landmark_year" %in% names(summary_df)) suppressWarnings(as.numeric(summary_df$landmark_year)) else rep(NA_real_, nrow(summary_df))
  tmp <- data.frame(landmark_year = yr, landmark_label = lab, stringsAsFactors = FALSE)

  year_map <- tmp %>%
    filter(is.finite(landmark_year)) %>%
    group_by(landmark_year) %>%
    summarise(landmark_label = pick_first_nonempty(landmark_label), .groups = "drop") %>%
    mutate(landmark_label = ifelse(is.na(landmark_label) | !nzchar(landmark_label), paste0("-", landmark_year, "yr"), landmark_label)) %>%
    arrange(desc(landmark_year))

  levels_main <- as.character(year_map$landmark_label)
  extra_levels <- unique(tmp$landmark_label[!is.finite(tmp$landmark_year) & nzchar(tmp$landmark_label)])
  extra_levels <- setdiff(extra_levels, levels_main)
  levels_main <- unique(c(levels_main, extra_levels))

  if (length(levels_main) == 0) {
    levels_main <- unique(tmp$landmark_label[nzchar(tmp$landmark_label)])
  }
  if (length(levels_main) == 0) stop("Cannot infer landmark labels from summary csv.")

  list(main = levels_main, forest = rev(levels_main))
}

calc_auc_limits <- function(summary_df) {
  auc <- suppressWarnings(as.numeric(summary_df$cindex_primary))
  lcl <- suppressWarnings(as.numeric(summary_df$cindex_primary_lcl))
  ucl <- suppressWarnings(as.numeric(summary_df$cindex_primary_ucl))
  finite_vals <- c(auc[is.finite(auc)], lcl[is.finite(lcl)], ucl[is.finite(ucl)])
  if (length(finite_vals) == 0) finite_vals <- c(0.45, 0.75)

  raw_min <- min(finite_vals, na.rm = TRUE)
  raw_max <- max(finite_vals, na.rm = TRUE)

  line_min <- max(0.2, min(0.5, raw_min - 0.03))
  line_max <- min(0.95, raw_max + 0.03)
  if (line_max - line_min < 0.12) {
    mid <- (line_min + line_max) / 2
    line_min <- max(0.2, mid - 0.06)
    line_max <- min(0.95, mid + 0.06)
  }

  noerr_min <- max(0.2, raw_min - 0.02)
  noerr_max <- min(0.95, raw_max + 0.02)
  if (noerr_max - noerr_min < 0.10) {
    mid <- (noerr_min + noerr_max) / 2
    noerr_min <- max(0.2, mid - 0.05)
    noerr_max <- min(0.95, mid + 0.05)
  }

  ci_min <- max(0.2, raw_min - 0.03)
  ci_max <- min(0.95, raw_max + 0.03)
  if (ci_max - ci_min < 0.12) {
    mid <- (ci_min + ci_max) / 2
    ci_min <- max(0.2, mid - 0.06)
    ci_max <- min(0.95, mid + 0.06)
  }

  list(
    line_min = line_min,
    line_max = line_max,
    noerr_min = noerr_min,
    noerr_max = noerr_max,
    ci_min = ci_min,
    ci_max = ci_max
  )
}

figure5_theme <- function(plot_family, base_size = 16, nonlegend_mult = NONLEGEND_TEXT_MULT) {
  theme_classic(base_family = plot_family, base_size = base_size) +
    theme(
      axis.title = element_text(size = base_size * nonlegend_mult, face = "bold", color = "black"),
      axis.text = element_text(size = base_size * 0.85 * nonlegend_mult, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.8),
      axis.ticks = element_line(color = "black", linewidth = 0.8),
      legend.title = element_text(size = base_size * 0.9, face = "bold"),
      legend.text = element_text(size = base_size * 0.8),
      strip.text = element_text(size = base_size * 0.9 * nonlegend_mult, face = "bold"),
      strip.background = element_blank(),
      panel.grid.major.y = element_line(color = "#E5E5E5", linewidth = 0.5),
      panel.grid.major.x = element_line(color = "#F0F0F0", linewidth = 0.35)
    )
}

save_plot_pdf_png <- function(plot_obj, fig_dir, stem, width, height, dpi = 300) {
  pdf_path <- file.path(fig_dir, paste0(stem, ".pdf"))
  png_path <- file.path(fig_dir, paste0(stem, ".png"))
  if (capabilities("cairo")) {
    ggsave(pdf_path, plot_obj, width = width, height = height, bg = "white", device = grDevices::cairo_pdf)
  } else {
    ggsave(pdf_path, plot_obj, width = width, height = height, bg = "white", device = grDevices::pdf, useDingbats = FALSE)
  }
  ggsave(png_path, plot_obj, width = width, height = height, dpi = dpi, bg = "white")
  c(pdf = pdf_path, png = png_path)
}

apply_auc_axis_break <- function(plot_obj, lower = 0.40, upper = 0.60, scales = 4.50, space = 0.01, symbol = "\u26A1") {
  if (!requireNamespace("ggbreak", quietly = TRUE)) return(plot_obj)
  plot_obj +
    ggbreak::scale_y_break(
      breaks = c(lower, upper),
      scales = scales,
      space = space,
      symbol = symbol
    ) +
    theme(
      axis.text.y.right = element_blank(),
      axis.ticks.y.right = element_blank(),
      axis.line.y.right = element_blank(),
      axis.title.y.right = element_blank()
    )
}

main <- function() {
  args <- parse_args()
  script_dir <- get_script_dir()
  res_dir <- resolve_path(args$results_dir, script_dir)
  fig_dir <- resolve_path(args$figures_dir, script_dir)
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  summary_load <- load_summary_csv(res_dir)
  summary_df <- summary_load$df
  if (nrow(summary_df) == 0) stop("Summary csv is empty in: ", res_dir)

  auc_req <- suppressWarnings(as.numeric(summary_df$cindex_primary_time_requested))
  auc_req <- auc_req[is.finite(auc_req) & auc_req > 0]
  primary_auc_time <- if (length(auc_req) > 0) auc_req[1] else 10

  landmark_levels <- infer_landmark_levels(summary_df)
  auc_limits <- calc_auc_limits(summary_df)
  plot_family <- resolve_plot_family("Arial")
  proteins_order <- unique(as.character(summary_df$protein))
  proteins_order <- proteins_order[nzchar(proteins_order)]
  if (length(proteins_order) == 0) proteins_order <- unique(as.character(summary_df$protein_col))
  proteins_order <- proteins_order[nzchar(proteins_order)]
  if (length(proteins_order) == 0) stop("Cannot infer protein names.")

  p5a <- summary_df %>%
    mutate(
      landmark_label = factor(landmark_label, levels = landmark_levels$main),
      time_flag = ifelse(is.finite(cindex_primary_time_used) & abs(cindex_primary_time_used - primary_auc_time) > 1e-8, "Adaptive time", "Requested time")
    ) %>%
    ggplot(aes(x = landmark_label, y = cindex_primary, color = protein, group = protein, shape = time_flag)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", linewidth = 1.0) +
    geom_errorbar(aes(ymin = cindex_primary_lcl, ymax = cindex_primary_ucl), width = 0.055, linewidth = 0.95, alpha = 0.70, na.rm = TRUE) +
    geom_line(linewidth = 1.1, na.rm = TRUE) +
    geom_point(size = 3.4, na.rm = TRUE) +
    scale_shape_manual(values = c("Requested time" = 16, "Adaptive time" = 17), drop = FALSE) +
    scale_y_continuous(
      limits = c(auc_limits$line_min, auc_limits$line_max),
      breaks = pretty(c(auc_limits$line_min, auc_limits$line_max), n = 6),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    figure5_theme(plot_family, base_size = 18) +
    labs(title = NULL, subtitle = NULL, x = "Landmark", y = "Time-dependent AUC", color = "Protein", shape = "Time source")

  p5a_alt <- summary_df %>%
    mutate(
      landmark_label = factor(landmark_label, levels = landmark_levels$main),
      protein = factor(protein, levels = proteins_order),
      time_flag = ifelse(is.finite(cindex_primary_time_used) & abs(cindex_primary_time_used - primary_auc_time) > 1e-8, "Adaptive time", "Requested time"),
      auc_label = ifelse(is.finite(cindex_primary), sprintf("%.3f%s", cindex_primary, ifelse(time_flag == "Adaptive time", "*", "")), "NA")
    ) %>%
    ggplot(aes(x = landmark_label, y = protein, fill = cindex_primary)) +
    geom_tile(color = "white", linewidth = 0.9) +
    geom_text(aes(label = auc_label), size = scale_nonlegend_text(4.6), family = plot_family, fontface = "bold", color = "black", na.rm = TRUE) +
    scale_fill_gradientn(
      colors = c("#3B4CC0", "#78A7FF", "#D8ECFF", "#FEE090", "#F46D43", "#A50026"),
      limits = c(auc_limits$line_min, auc_limits$line_max),
      name = "Time-dependent AUC",
      na.value = "#F2F2F2"
    ) +
    theme_minimal(base_family = plot_family, base_size = 16) +
    theme(
      axis.title = element_text(size = scale_nonlegend_text(16), face = "bold", color = "black"),
      axis.text = element_text(size = scale_nonlegend_text(13), color = "black"),
      panel.grid = element_blank(),
      legend.title = element_text(size = 13, face = "bold"),
      legend.text = element_text(size = 12),
      plot.caption = element_text(size = scale_nonlegend_text(11), hjust = 0, color = "#555555")
    ) +
    labs(
      title = NULL,
      subtitle = NULL,
      x = "Landmark",
      y = "Protein",
      caption = "* Adaptive time used for this cell"
    )

  p5a_alt_ci <- summary_df %>%
    mutate(
      landmark_label = factor(landmark_label, levels = landmark_levels$main),
      protein = factor(protein, levels = rev(proteins_order)),
      time_flag = ifelse(is.finite(cindex_primary_time_used) & abs(cindex_primary_time_used - primary_auc_time) > 1e-8, "Adaptive time", "Requested time"),
      auc_text = ifelse(
        is.finite(cindex_primary),
        paste0(sprintf("%.3f", cindex_primary), " (", sprintf("%.3f", cindex_primary_lcl), "-", sprintf("%.3f", cindex_primary_ucl), ")\nP=", fmt_p(cindex_primary_p)),
        "NA"
      )
    )
  p5a_alt_ci <- p5a_alt_ci %>%
    ggplot(aes(y = protein, x = cindex_primary, color = protein, shape = time_flag)) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = "red", linewidth = 0.95) +
    geom_errorbarh(aes(xmin = cindex_primary_lcl, xmax = cindex_primary_ucl), height = 0.20, linewidth = 1.0, na.rm = TRUE) +
    geom_point(size = 3.6, na.rm = TRUE) +
    geom_text(aes(label = auc_text), hjust = 0.5, nudge_y = -0.23, size = scale_nonlegend_text(3.2), lineheight = 0.92, family = plot_family, color = "black", show.legend = FALSE, na.rm = TRUE) +
    facet_wrap(~landmark_label, nrow = 1) +
    coord_cartesian(xlim = c(auc_limits$ci_min, auc_limits$ci_max), clip = "off") +
    scale_shape_manual(values = c("Requested time" = 16, "Adaptive time" = 17), drop = FALSE) +
    scale_y_discrete(expand = expansion(mult = c(0.12, 0.28))) +
    theme_classic(base_family = plot_family, base_size = 15) +
    theme(
      axis.title = element_text(size = scale_nonlegend_text(15), face = "bold"),
      axis.text = element_text(size = scale_nonlegend_text(12)),
      strip.text = element_text(size = scale_nonlegend_text(13), face = "bold"),
      legend.title = element_text(size = 13, face = "bold"),
      legend.text = element_text(size = 12),
      plot.margin = margin(10, 26, 20, 10)
    ) +
    labs(title = NULL, subtitle = NULL, x = "Time-dependent AUC", y = "Protein", color = "Protein", shape = "Time source")

  noerr_span <- auc_limits$noerr_max - auc_limits$noerr_min
  top_ylim_min <- 0.30
  top_ylim_max <- 0.79
  top_auc_breaks <- c(0.30, 0.40, 0.60, 0.70)
  crowd_gap <- 0.012
  base_drop <- 0.028 * (top_ylim_max - top_ylim_min)
  stack_step <- 0.020 * (top_ylim_max - top_ylim_min)
  x_step <- 0.09
  max_x_step <- 0.27
  label_pad <- 0.04 * (top_ylim_max - top_ylim_min)
  if (!is.finite(label_pad) || label_pad <= 0) label_pad <- 0.01

  p5a_noerr_df <- summary_df %>%
    mutate(
      landmark_label = factor(landmark_label, levels = landmark_levels$main),
      landmark_num = as.integer(landmark_label),
      protein = factor(protein, levels = proteins_order),
      time_flag = ifelse(is.finite(cindex_primary_time_used) & abs(cindex_primary_time_used - primary_auc_time) > 1e-8, "Adaptive time", "Requested time"),
      auc_text = ifelse(
        is.finite(cindex_primary),
        paste0(sprintf("%.3f", cindex_primary), " (", sprintf("%.3f", cindex_primary_lcl), "-", sprintf("%.3f", cindex_primary_ucl), "), P=", fmt_p(cindex_primary_p)),
        "NA"
      ),
      auc_text_star = ifelse(
        is.finite(cindex_primary),
        paste0(sprintf("%.3f", cindex_primary), " (", sprintf("%.3f", cindex_primary_lcl), "-", sprintf("%.3f", cindex_primary_ucl), ")", sig_star(cindex_primary_p)),
        "NA"
      )
    ) %>%
    group_by(landmark_num) %>%
    arrange(desc(cindex_primary), .by_group = TRUE) %>%
    mutate(
      gap_prev = lag(cindex_primary) - cindex_primary,
      crowd_chain = cumsum(ifelse(is.na(gap_prev) | gap_prev >= crowd_gap, 1L, 0L))
    ) %>%
    group_by(landmark_num, crowd_chain) %>%
    mutate(
      chain_n = n(),
      chain_idx = row_number(),
      crowd_slot = ifelse(chain_idx == 1, 0, ceiling((chain_idx - 1) / 2) * ifelse(chain_idx %% 2 == 0, -1, 1)),
      label_dx = pmax(-max_x_step, pmin(max_x_step, crowd_slot * x_step)),
      label_hjust = case_when(
        label_dx < 0 ~ 1,
        label_dx > 0 ~ 0,
        TRUE ~ 0.5
      ),
      label_drop = base_drop + (chain_idx - 1) * stack_step,
      label_x = landmark_num + label_dx,
      label_y = pmin(top_ylim_max - label_pad, pmax(top_ylim_min + label_pad, cindex_primary - label_drop))
    ) %>%
    ungroup()

  top_sig_note_label <- "P < 0.01: *"
  top_sig_legend_df <- data.frame(
    x = seq_along(landmark_levels$main)[1],
    xend = seq_along(landmark_levels$main)[1],
    y = top_ylim_max,
    yend = top_ylim_max,
    sig_note = top_sig_note_label,
    stringsAsFactors = FALSE
  )

  p5a_alt_noerr <- p5a_noerr_df %>%
    ggplot(aes(x = landmark_num, y = cindex_primary, color = protein, group = protein, shape = time_flag)) +
    geom_line(linewidth = 0.95, show.legend = FALSE, na.rm = TRUE) +
    geom_point(size = 3.2, na.rm = TRUE) +
    geom_segment(
      aes(xend = label_x, yend = label_y),
      color = "#C8C8C8",
      linewidth = 0.55,
      alpha = 0.95,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    geom_text(aes(x = label_x, y = label_y, label = auc_text, hjust = label_hjust), vjust = 0.5, size = scale_nonlegend_text(3.2), lineheight = 0.92, family = plot_family, color = "black", show.legend = FALSE, na.rm = TRUE) +
    scale_x_continuous(
      breaks = seq_along(landmark_levels$main),
      labels = landmark_levels$main,
      expand = expansion(mult = c(0.14, 0.14))
    ) +
    scale_shape_manual(values = c("Requested time" = 16, "Adaptive time" = 17), drop = FALSE) +
    scale_y_continuous(
      limits = c(top_ylim_min, top_ylim_max),
      breaks = top_auc_breaks,
      expand = expansion(mult = c(0.00, 0.00))
    ) +
    figure5_theme(plot_family, base_size = 16) +
    theme(
      axis.text.x = element_text(size = scale_nonlegend_text(16.32)),
      axis.text.y = element_text(size = scale_nonlegend_text(16.32)),
      plot.margin = margin(10, 12, 18, 10)
    ) +
    labs(title = NULL, subtitle = NULL, x = "Landmark", y = "Time-dependent AUC", color = "Protein", shape = "Time source")
  p5a_alt_noerr <- apply_auc_axis_break(p5a_alt_noerr)

  p5a_alt_noerr_sigstar <- p5a_noerr_df %>%
    ggplot(aes(x = landmark_num, y = cindex_primary, color = protein, group = protein, shape = time_flag)) +
    geom_line(linewidth = 0.95, show.legend = FALSE, na.rm = TRUE) +
    geom_point(size = 3.2, na.rm = TRUE) +
    geom_segment(
      data = top_sig_legend_df,
      aes(x = x, xend = xend, y = y, yend = yend, linetype = sig_note),
      inherit.aes = FALSE,
      color = "black",
      alpha = 0,
      linewidth = 0.9,
      show.legend = TRUE
    ) +
    geom_segment(
      aes(xend = label_x, yend = label_y),
      color = "#C8C8C8",
      linewidth = 0.55,
      alpha = 0.95,
      show.legend = FALSE,
      na.rm = TRUE
    ) +
    geom_text(aes(x = label_x, y = label_y, label = auc_text_star, hjust = label_hjust), vjust = 0.5, size = scale_nonlegend_text(3.2), lineheight = 0.92, family = plot_family, color = "black", show.legend = FALSE, na.rm = TRUE) +
    scale_x_continuous(
      breaks = seq_along(landmark_levels$main),
      labels = landmark_levels$main,
      expand = expansion(mult = c(0.14, 0.14))
    ) +
    scale_shape_manual(values = c("Requested time" = 16, "Adaptive time" = 17), drop = FALSE) +
    scale_linetype_manual(
      name = "P-value",
      values = c("P < 0.01: *" = "solid"),
      guide = guide_legend(
        order = 3,
        keywidth = grid::unit(0, "pt"),
        keyheight = grid::unit(0, "pt"),
        label.hjust = 0,
        override.aes = list(linetype = 0, alpha = 0, linewidth = 0, color = NA)
      )
    ) +
    scale_y_continuous(
      limits = c(top_ylim_min, top_ylim_max),
      breaks = top_auc_breaks,
      expand = expansion(mult = c(0.00, 0.00))
    ) +
    figure5_theme(plot_family, base_size = 16) +
    guides(shape = guide_legend(order = 1), color = guide_legend(order = 2)) +
    theme(
      axis.text.x = element_text(size = scale_nonlegend_text(16.32)),
      axis.text.y = element_text(size = scale_nonlegend_text(16.32)),
      plot.margin = margin(10, 12, 18, 10)
    ) +
    labs(title = NULL, subtitle = NULL, x = "Landmark", y = "Time-dependent AUC", color = "Protein", shape = "Time source")
  p5a_alt_noerr_sigstar <- apply_auc_axis_break(p5a_alt_noerr_sigstar)

  p5b <- NULL
  p5b_sigstar <- NULL
  if (!isTRUE(args$no_forest)) {
    p5b_df <- summary_df %>%
      mutate(landmark_label = factor(landmark_label, levels = landmark_levels$forest)) %>%
      filter(is.finite(sub_hr), is.finite(sub_hr_lcl), is.finite(sub_hr_ucl), sub_hr > 0, sub_hr_lcl > 0, sub_hr_ucl > 0)

    if (nrow(p5b_df) == 0) {
      p5b <- ggplot() +
        annotate("text", x = 1, y = 1, label = "No finite HR results for Figure 5B", size = scale_nonlegend_text(7), family = plot_family) +
        theme_void()
      p5b_sigstar <- p5b
    } else {
      xmin <- 0.5
      xmax <- min(8, max(p5b_df$sub_hr_ucl, na.rm = TRUE) * 1.35)
      if (!is.finite(xmax) || xmax <= xmin) xmax <- xmin * 1.8
      x_breaks <- c(0.5, 1.0, 2.0)
      x_breaks <- x_breaks[x_breaks >= xmin & x_breaks <= xmax]
      if (length(x_breaks) == 0) x_breaks <- c(xmin)
      right_label_proteins <- c("BCAN")
      p5b <- p5b_df %>%
        mutate(
          protein = factor(protein, levels = proteins_order),
          protein_chr = as.character(protein),
          label_right = protein_chr %in% right_label_proteins,
          txt_x = ifelse(label_right, pmin(xmax * 0.96, pmax(0.95, sub_hr * 1.35)), sub_hr),
          txt = ifelse(
            is.finite(sub_hr),
            paste0(sprintf("%.2f", sub_hr), " (", sprintf("%.2f", sub_hr_lcl), "-", sprintf("%.2f", sub_hr_ucl), ")\nP=", fmt_p(sub_hr_p)),
            "NA"
          ),
          txt_star = ifelse(
            is.finite(sub_hr),
            paste0(sprintf("%.2f", sub_hr), " (", sprintf("%.2f", sub_hr_lcl), "-", sprintf("%.2f", sub_hr_ucl), ")", sig_star(sub_hr_p)),
            "NA"
          )
        ) %>%
        ggplot(aes(y = landmark_label, x = sub_hr, color = landmark_label)) +
        geom_vline(xintercept = 1, linetype = "dashed", color = "red", linewidth = 1.0) +
        geom_errorbarh(aes(xmin = sub_hr_lcl, xmax = sub_hr_ucl), height = 0.22, linewidth = 2.0, na.rm = TRUE) +
        geom_point(size = 6.4, na.rm = TRUE) +
        geom_text(
          data = function(d) d %>% filter(!label_right),
          aes(x = sub_hr, label = txt),
          hjust = 0.5,
          nudge_y = -0.42,
          size = scale_nonlegend_text(3.6),
          lineheight = 0.92,
          family = plot_family,
          color = "black"
        ) +
        geom_text(
          data = function(d) d %>% filter(label_right),
          aes(x = txt_x, label = txt),
          hjust = 0,
          vjust = 0.5,
          size = scale_nonlegend_text(3.6),
          lineheight = 0.92,
          family = plot_family,
          color = "black"
        ) +
        scale_x_log10(limits = c(xmin, xmax), breaks = x_breaks, labels = function(x) sprintf("%.1f", x)) +
        scale_y_discrete(expand = expansion(mult = c(0.78, 0.18))) +
        facet_wrap(~protein, ncol = 4, scales = "free_y") +
        coord_cartesian(clip = "off") +
        figure5_theme(plot_family, base_size = 16) +
        theme(
          axis.text.x = element_text(size = scale_nonlegend_text(16.32)),
          axis.text.y = element_text(size = scale_nonlegend_text(16.32)),
          plot.margin = margin(6, 8, 26, 6)
        ) +
        labs(title = NULL, subtitle = NULL, x = "Subdistribution HR (log scale)", y = "Landmark", color = "Landmark")

      sig_note_label <- "P < 0.01: *"
      sig_legend_df <- data.frame(
        sub_hr = xmin,
        landmark_label = factor(landmark_levels$forest[1], levels = landmark_levels$forest),
        protein = factor(proteins_order[1], levels = proteins_order),
        sig_note = sig_note_label,
        stringsAsFactors = FALSE
      )

      p5b_sigstar <- p5b_df %>%
        mutate(
          protein = factor(protein, levels = proteins_order),
          protein_chr = as.character(protein),
          label_right = protein_chr %in% right_label_proteins,
          txt_x = ifelse(label_right, pmin(xmax * 0.96, pmax(0.95, sub_hr * 1.35)), sub_hr),
          txt_star = ifelse(
            is.finite(sub_hr),
            paste0(sprintf("%.2f", sub_hr), " (", sprintf("%.2f", sub_hr_lcl), "-", sprintf("%.2f", sub_hr_ucl), ")", sig_star(sub_hr_p)),
            "NA"
          )
        ) %>%
        ggplot(aes(y = landmark_label, x = sub_hr, color = landmark_label)) +
        geom_vline(xintercept = 1, linetype = "dashed", color = "red", linewidth = 1.0) +
        geom_errorbarh(aes(xmin = sub_hr_lcl, xmax = sub_hr_ucl), height = 0.22, linewidth = 2.0, na.rm = TRUE) +
        geom_point(size = 6.4, na.rm = TRUE) +
        geom_point(
          data = sig_legend_df,
          aes(x = sub_hr, y = landmark_label, shape = sig_note),
          inherit.aes = FALSE,
          color = "black",
          alpha = 0,
          size = 4.2,
          show.legend = TRUE
        ) +
        geom_text(
          data = function(d) d %>% filter(!label_right),
          aes(x = sub_hr, label = txt_star),
          hjust = 0.5,
          nudge_y = -0.42,
          size = scale_nonlegend_text(3.6),
          lineheight = 0.92,
          family = plot_family,
          color = "black"
        ) +
        geom_text(
          data = function(d) d %>% filter(label_right),
          aes(x = txt_x, label = txt_star),
          hjust = 0,
          vjust = 0.5,
          size = scale_nonlegend_text(3.6),
          lineheight = 0.92,
          family = plot_family,
          color = "black"
        ) +
        scale_x_log10(limits = c(xmin, xmax), breaks = x_breaks, labels = function(x) sprintf("%.1f", x)) +
        scale_y_discrete(expand = expansion(mult = c(0.78, 0.18))) +
        scale_shape_manual(
          name = "P-value",
          values = c("P < 0.01: *" = 8),
          guide = guide_legend(
            order = 2,
            keywidth = grid::unit(0, "pt"),
            keyheight = grid::unit(0, "pt"),
            label.hjust = 0,
            override.aes = list(shape = NA, alpha = 0, size = 0, color = NA)
          )
        ) +
        facet_wrap(~protein, ncol = 4, scales = "free_y") +
        coord_cartesian(clip = "off") +
        figure5_theme(plot_family, base_size = 16) +
        guides(color = guide_legend(order = 1)) +
        theme(
          axis.text.x = element_text(size = scale_nonlegend_text(16.32)),
          axis.text.y = element_text(size = scale_nonlegend_text(16.32)),
          plot.margin = margin(6, 8, 26, 6)
        ) +
        labs(title = NULL, subtitle = NULL, x = "Subdistribution HR (log scale)", y = "Landmark", color = "Landmark")
    }
  }

  suffix <- trimws(args$figure_suffix)
  use_suffix4 <- isTRUE(args$figure54_names)

  stem_top <- if (use_suffix4) "Figure5A4_landmark_multi_proteins_alt_noerr_labels_sigstar" else "Figure5A_landmark_multi_proteins_alt_noerr_labels_sigstar"
  stem_bottom <- if (use_suffix4) "Figure5B4_landmark_forest_sigstar" else "Figure5B_landmark_forest_sigstar"
  stem_top <- paste0(stem_top, suffix)
  stem_bottom <- paste0(stem_bottom, suffix)

  top_bottom_width_in <- 24.0
  top_bottom_height_in <- 10.0

  out_top <- save_plot_pdf_png(p5a_alt_noerr_sigstar, fig_dir, stem_top, width = top_bottom_width_in, height = top_bottom_height_in)
  out_bottom <- NULL
  if (!isTRUE(args$no_forest)) {
    out_bottom <- save_plot_pdf_png(p5b_sigstar, fig_dir, stem_bottom, width = top_bottom_width_in, height = top_bottom_height_in)
  }

  cat("[Done] Saved only top and bottom figures:\n")
  cat("  summary source(s):\n")
  for (src in summary_load$sources) cat("    - ", normalizePath(src, winslash = "/", mustWork = FALSE), "\n", sep = "")
  cat("  top   : ", normalizePath(out_top["png"], winslash = "/", mustWork = FALSE), "\n", sep = "")
  if (!isTRUE(args$no_forest)) {
    cat("  bottom: ", normalizePath(out_bottom["png"], winslash = "/", mustWork = FALSE), "\n", sep = "")
  }
}

if (identical(environment(), globalenv())) main()
