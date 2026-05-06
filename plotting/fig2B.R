suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(patchwork)
  library(grid)
  library(png)
})

# ===== Input / Output =====
csv_path <- "pro5317_morbidity_delphi_mri_no_baseline_cvd_cmr19.csv"
out_pdf <- "cmr_dotplot_only_rot90_plus180_13proteins_len24.pdf"
out_png <- "cmr_dotplot_only_rot90_plus180_13proteins_len24.png"
top_n_proteins <- 30
rotation_angle <- 270
target_pdf_width <- 24

target_proteins <- c(
  "ACTA2", "CXCL17", "MMP12", "CDCP1", "BCAN", "WFDC2", "GDF15",
  "EDA2R", "NTproBNP", "CDHR2", "RBFOX3", "HAVCR1", "HSPB6"
)

normalize_token <- function(x) {
  gsub("[^A-Z0-9]", "", toupper(as.character(x)))
}

if (!file.exists(csv_path)) {
  stop("Input CSV not found in current folder: ", normalizePath(csv_path, mustWork = FALSE))
}

# ===== Read and preprocess =====
df <- readr::read_csv(csv_path, show_col_types = FALSE)

required_cols <- c("Protein", "Outcome", "beta", "SE")
if (!all(required_cols %in% names(df))) {
  stop("CSV must contain: ", paste(required_cols, collapse = ", "))
}

pcol <- if ("p_adj" %in% names(df)) "p_adj" else if ("P_value" %in% names(df)) "P_value" else NA_character_
if (is.na(pcol)) {
  stop("CSV must contain either p_adj or P_value column.")
}

la_set <- c("LA ejection fraction", "LA maximum volume", "LA minimum volume", "LA stroke volume")
lv_set <- c(
  "LV cardiac output", "LV ejection fraction", "LV end diastolic volume", "LV end systolic volume",
  "LV mean myocardial wall thickness global", "LV myocardial mass", "LV stroke volume"
)
ra_set <- c("RA ejection fraction", "RA maximum volume", "RA minimum volume", "RA stroke volume")
rv_set <- c("RV ejection fraction", "RV end diastolic volume", "RV end systolic volume", "RV stroke volume")

df <- df %>%
  mutate(
    q = pmax(.data[[pcol]], .Machine$double.xmin),
    neglog10 = -log10(q),
    sign = ifelse(beta >= 0, "pos", "neg"),
    Chamber = case_when(
      Outcome %in% la_set ~ "Left Atrium (LA)",
      Outcome %in% lv_set ~ "Left Ventricle (LV)",
      Outcome %in% ra_set ~ "Right Atrium (RA)",
      Outcome %in% rv_set ~ "Right Ventricle (RV)",
      TRUE ~ "Other"
    )
  ) %>%
  filter(Chamber != "Other") %>%
  mutate(
    Outcome = factor(Outcome, levels = c(la_set, lv_set, ra_set, rv_set)),
    Chamber = factor(Chamber, levels = c("Left Atrium (LA)", "Left Ventricle (LV)", "Right Atrium (RA)", "Right Ventricle (RV)")),
    ChamberShort = case_when(
      Chamber == "Left Atrium (LA)" ~ "LA",
      Chamber == "Left Ventricle (LV)" ~ "LV",
      Chamber == "Right Atrium (RA)" ~ "RA",
      Chamber == "Right Ventricle (RV)" ~ "RV",
      TRUE ~ NA_character_
    ),
    ChamberShort = factor(ChamberShort, levels = c("LA", "LV", "RA", "RV"))
  )

top_prots <- df %>%
  group_by(Protein) %>%
  summarise(pmin = min(q, na.rm = TRUE), .groups = "drop") %>%
  arrange(pmin) %>%
  slice_head(n = top_n_proteins) %>%
  pull(Protein)

target_lookup <- setNames(target_proteins, normalize_token(target_proteins))
protein_map <- df %>%
  distinct(Protein) %>%
  mutate(norm = normalize_token(Protein))

forced_proteins <- protein_map %>%
  filter(norm %in% names(target_lookup)) %>%
  pull(Protein)

missing_targets <- setdiff(names(target_lookup), normalize_token(forced_proteins))
if (length(missing_targets) > 0) {
  warning("Targets not found in CSV: ", paste(target_lookup[missing_targets], collapse = ", "))
}

selected_proteins <- unique(c(top_prots, forced_proteins))
df_top <- df %>% filter(Protein %in% selected_proteins)

prot_order <- df_top %>%
  group_by(Protein) %>%
  summarise(pmin = min(q, na.rm = TRUE), .groups = "drop") %>%
  arrange(pmin) %>%
  pull(Protein)

df_top <- df_top %>%
  mutate(
    Protein = factor(Protein, levels = rev(prot_order))
  )

# ===== Colors =====
pal_pos <- "#E64B35FF"
pal_neg <- "#4DBBD5FF"
chamber_cols <- c(
  LA = "#8DA0CB",
  LV = "#FC8D62",
  RA = "#66C2A5",
  RV = "#E78AC3"
)

# ===== Build plot (strip + bubble heatmap) =====
strip_df <- df_top %>% distinct(Outcome, ChamberShort)

p_strip <- ggplot(strip_df, aes(x = Outcome, y = 1, fill = ChamberShort)) +
  geom_tile(height = 1) +
  scale_fill_manual(
    values = chamber_cols,
    name = "Chamber",
    breaks = names(chamber_cols),
    labels = names(chamber_cols),
    guide = guide_legend(order = 1, keyheight = unit(5, "mm"))
  ) +
  scale_y_continuous(expand = expansion(c(0, 0))) +
  theme_void(base_family = "sans", base_size = 11) +
  theme(
    legend.position = "right",
    plot.margin = margin(t = 5, r = 10, b = 0, l = 10)
  )

p_heat <- ggplot(df_top, aes(x = Outcome, y = Protein)) +
  geom_tile(aes(fill = beta), width = 0.9, height = 0.9, alpha = 0.92) +
  geom_point(aes(size = neglog10, color = sign), alpha = 0.9) +
  scale_fill_gradient2(
    low = pal_neg,
    mid = "#FFFFFF",
    high = pal_pos,
    midpoint = 0,
    name = "Beta",
    guide = guide_colorbar(order = 2)
  ) +
  scale_color_manual(values = c(pos = pal_pos, neg = pal_neg), guide = "none") +
  scale_size_continuous(name = expression(-log[10](q)), range = c(0.5, 3.8)) +
  scale_x_discrete(position = "bottom") +
  theme_classic(base_size = 16) +
  theme(
    text = element_text(color = "black"),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(angle = -90, hjust = 0, vjust = 0.5, size = 24, color = "black"),
    axis.text.y = element_text(size = 24, color = "black"),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    plot.margin = margin(t = 0, r = 10, b = 10, l = 10)
  )

combined <- (p_strip / p_heat) +
  plot_layout(heights = c(0.3, 8.8), guides = "collect") &
  theme(
    legend.position = "right",
    text = element_text(family = "sans", color = "black"),
    legend.text = element_text(size = 16, color = "black", angle = 0),
    legend.title = element_text(size = 16, color = "black", angle = 0)
  )

rotate_png <- function(img, angle = 0) {
  if (length(dim(img)) != 3) {
    stop("Image array must be H x W x C.")
  }
  angle <- angle %% 360
  if (angle == 0) return(img)
  if (angle == 90) {
    return(aperm(img, c(2, 1, 3))[, nrow(img):1, , drop = FALSE])
  }
  if (angle == 180) {
    return(img[nrow(img):1, ncol(img):1, , drop = FALSE])
  }
  if (angle == 270) {
    return(aperm(img, c(2, 1, 3))[ncol(img):1, , , drop = FALSE])
  }
  stop("angle must be one of 0, 90, 180, 270.")
}

# ===== Save only this figure (rotated, no clipping) =====
tmp_png <- tempfile(fileext = ".png")
ggsave(tmp_png, combined, width = 10, height = 24, dpi = 300, bg = "white")

img <- png::readPNG(tmp_png)
img_rot <- rotate_png(img, angle = rotation_angle)
png::writePNG(img_rot, target = out_png)

img_h <- dim(img_rot)[1]
img_w <- dim(img_rot)[2]
pdf_w <- target_pdf_width
pdf_h <- target_pdf_width * img_h / img_w

cairo_pdf(out_pdf, width = pdf_w, height = pdf_h)
grid.newpage()
grid.raster(img_rot, x = 0.5, y = 0.5, width = unit(1, "npc"), height = unit(1, "npc"), interpolate = TRUE)
dev.off()

unlink(tmp_png)

message("Done. Saved:")
message("  - ", normalizePath(out_pdf, winslash = "/", mustWork = FALSE))
message("  - ", normalizePath(out_png, winslash = "/", mustWork = FALSE))
