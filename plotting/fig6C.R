#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(stringr)
})

input_dir <- "TableC_Mediation_Results_BMI"
output_pdf <- "mediation_circular_plot_BMI_radius_clean_fullbase.pdf"
top_k <- 3
empty_bar <- 1

if (!dir.exists(input_dir)) {
  stop("Input directory not found: ", normalizePath(input_dir, mustWork = FALSE))
}

files <- list.files(input_dir, pattern = "^TableC_.*_Mediation\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  stop("No mediation CSV files found in: ", normalizePath(input_dir, mustWork = FALSE))
}

read_one <- function(fp) {
  disease <- str_match(basename(fp), "^TableC_([A-Za-z0-9]+)_Mediation\\.csv$")[, 2]
  df <- read_csv(fp, show_col_types = FALSE)
  if (!all(c("protein", "PM", "PM_p") %in% names(df))) {
    return(tibble())
  }
  df %>%
    transmute(
      protein = as.character(protein),
      PM = suppressWarnings(as.numeric(PM)),
      PM_p = suppressWarnings(as.numeric(PM_p)),
      Disease = disease
    )
}

raw <- bind_rows(lapply(files, read_one))
if (nrow(raw) == 0) {
  stop("No valid rows parsed from BMI mediation CSV files.")
}

plot_df <- raw %>%
  filter(!is.na(PM), !is.na(PM_p), PM_p < 0.05) %>%
  group_by(Disease) %>%
  arrange(desc(PM), .by_group = TRUE) %>%
  slice_head(n = top_k) %>%
  ungroup() %>%
  mutate(PM_pct = PM * 100)

if (nrow(plot_df) == 0) {
  stop("No rows left after filtering PM_p < 0.05 and top-k selection.")
}

# Keep disease order stable and close to previous plots
disease_order_ref <- c(
  "G45", "I05", "I07", "I08", "I10", "I12", "I20", "I21",
  "I25", "I26", "I27", "I35", "I37", "I42", "I44", "I45",
  "I46", "I48", "I50", "I51", "I63", "I67", "I69", "I71",
  "I74", "I77", "I78"
)
disease_levels <- unique(c(disease_order_ref, sort(unique(plot_df$Disease))))
disease_levels <- disease_levels[disease_levels %in% unique(plot_df$Disease)]

plot_df <- plot_df %>%
  mutate(Disease = factor(Disease, levels = disease_levels)) %>%
  arrange(Disease, desc(PM_pct))

# Insert one empty bar between diseases
gap_df <- tibble(
  protein = NA_character_,
  PM = NA_real_,
  PM_p = NA_real_,
  Disease = factor(rep(disease_levels, each = empty_bar), levels = disease_levels),
  PM_pct = NA_real_
)

plot_df_gap <- bind_rows(plot_df, gap_df) %>%
  arrange(Disease, desc(PM_pct))
plot_df_gap$id <- seq_len(nrow(plot_df_gap))

max_y <- ceiling(max(plot_df_gap$PM_pct, na.rm = TRUE) / 10) * 10
if (!is.finite(max_y) || max_y <= 0) max_y <- 100

# Disease colors
base_colors <- c(
  "G45" = "#1f77b4", "I05" = "#aec7e8", "I07" = "#ff7f0e", "I08" = "#ffbb78",
  "I10" = "#2ca02c", "I12" = "#98df8a", "I20" = "#d62728", "I21" = "#ff9896",
  "I25" = "#9467bd", "I26" = "#c5b0d5", "I27" = "#8c564b", "I35" = "#7f7f7f",
  "I37" = "#bcbd22", "I42" = "#dbdb8d", "I44" = "#17becf", "I45" = "#9edae5",
  "I46" = "#393b79", "I48" = "#6b6ecf", "I50" = "#637939", "I51" = "#8ca252",
  "I63" = "#b5cf6b", "I67" = "#8c6d31", "I69" = "#bd9e39", "I71" = "#e7cb94",
  "I74" = "#ad494a", "I77" = "#d6616b", "I78" = "#e7969c"
)
missing_cols <- setdiff(disease_levels, names(base_colors))
if (length(missing_cols) > 0) {
  fallback <- grDevices::hcl.colors(length(missing_cols), palette = "Dark 3")
  names(fallback) <- missing_cols
  base_colors <- c(base_colors, fallback)
}
pal <- base_colors[disease_levels]

main_df <- plot_df_gap %>% filter(!is.na(protein))
n_bar <- nrow(plot_df_gap)

# Outer percent labels
pct_labels <- main_df %>%
  mutate(
    label = sprintf("%.1f%%", PM_pct),
    angle = 90 - 360 * (id - 0.5) / n_bar,
    hjust = ifelse(angle < -90, 1, 0),
    angle = ifelse(angle < -90, angle + 180, angle),
    y = PM_pct + max(2, 0.03 * max_y)
  )

# Protein labels: keep every bar label (no dedup), then lightly stagger near bar base
prot_labels <- main_df %>%
  transmute(protein, id, theta = 360 * (id - 0.5) / n_bar) %>%
  arrange(theta)

assign_levels <- function(theta_deg, min_sep = 7) {
  n <- length(theta_deg)
  lev <- integer(n)
  if (n <= 1) return(lev)
  for (i in seq_len(n)) {
    if (i == 1) {
      lev[i] <- 0
      next
    }
    prev <- seq_len(i - 1)
    d <- abs(theta_deg[i] - theta_deg[prev])
    circ_d <- pmin(d, 360 - d)
    used <- lev[prev][circ_d < min_sep]
    lvl <- 0
    while (lvl %in% used) lvl <- lvl + 1
    lev[i] <- lvl
  }
  lev %% 3
}

prot_labels$level <- assign_levels(prot_labels$theta, min_sep = 7)
inner_hole <- max_y * 0.85
base_y <- -inner_hole * 0.03
step_y <- max(1.0, inner_hole * 0.02)

prot_labels <- prot_labels %>%
  mutate(
    angle = 90 - 360 * (id - 0.5) / n_bar,
    hjust = ifelse(angle < -90, 1, 0),
    angle = ifelse(angle < -90, angle + 180, angle),
    y = base_y - level * step_y
  )

p <- ggplot(plot_df_gap, aes(x = factor(id), y = PM_pct, fill = Disease)) +
  geom_col(width = 0.95, color = "white", alpha = 0.96, na.rm = TRUE) +
  coord_polar() +
  scale_fill_manual(values = pal, drop = FALSE) +
  scale_y_continuous(limits = c(-inner_hole, max_y * 1.28), expand = c(0, 0)) +
  theme_void(base_size = 14) +
  theme(
    text = element_text(color = "black"),
    legend.title = element_text(size = 22, face = "bold"),
    legend.text = element_text(size = 18),
    plot.margin = margin(8, 8, 8, 8)
  ) +
  guides(
    fill = guide_legend(
      title = "Disease",
      ncol = 2,
      byrow = TRUE,
      keywidth = grid::unit(1.5, "lines"),
      keyheight = grid::unit(1.5, "lines")
    )
  )

p <- p +
  geom_text(
    data = pct_labels,
    aes(x = factor(id), y = y, label = label, angle = angle, hjust = hjust),
    inherit.aes = FALSE,
    size = 3.5,
    color = "black"
  ) +
  geom_text(
    data = prot_labels,
    aes(x = factor(id), y = y, label = protein, angle = angle, hjust = hjust),
    inherit.aes = FALSE,
    size = 3.6,
    color = "black"
  )

ggsave(output_pdf, p, device = cairo_pdf, width = 14, height = 10, bg = "white")
message("Saved: ", normalizePath(output_pdf, winslash = "/", mustWork = FALSE))
