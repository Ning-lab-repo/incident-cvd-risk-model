library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(scales)
library(ggrepel)

# ---------------------------
# Parameters
# ---------------------------
input_csv <- "cox_incident_results.csv"
use_bonferroni <- TRUE
p_threshold <- 0.05

barplot_file_pdf <- "significant_counts_by_disease_13proteins_highlight.pdf"
hr_plot_pdf <- "hr_manhattan_plot_by_disease_13proteins_highlight.pdf"

diseases_to_plot <- c(
  "I10", "I50", "I25", "I73", "I48", "I27", "I12", "I21", "I70", "I65",
  "I77", "I51", "I46", "I67", "I63", "I20", "I35", "I74", "I69", "I71",
  "I26", "I08", "I34", "I44", "I05", "I47", "I37", "I45", "I78", "I07",
  "I42", "G45", "I49", "I31", "I33", "I36"
)

highlight_targets <- c(
  "ACTA2", "CXCL17", "MMP12", "CDCP1", "BCAN", "WFDC2", "GDF15",
  "EDA2R", "NTproBNP", "CDHR2", "RBFOX3", "HAVCR1", "HSPB6"
)

normalize_token <- function(x) {
  gsub("[^A-Z0-9]", "", toupper(as.character(x)))
}

highlight_lookup <- setNames(highlight_targets, normalize_token(highlight_targets))

# ---------------------------
# Read and clean data
# ---------------------------
message("Reading input file: ", input_csv)
df_raw <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c("Exposure", "Outcome", "Hazard ratio (95%CI)", "P value", "p_bonferroni")
missing_cols <- setdiff(required_cols, colnames(df_raw))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df <- df_raw %>%
  mutate(
    HR = as.numeric(stringr::str_extract(`Hazard ratio (95%CI)`, "^[0-9]+\\.?[0-9]*")),
    logHR = ifelse(!is.na(HR) & HR > 0, log(HR), NA_real_),
    P_value = as.numeric(`P value`),
    p_bonferroni = as.numeric(p_bonferroni),
    is_signif = if (use_bonferroni) {
      !is.na(p_bonferroni) & p_bonferroni < p_threshold
    } else {
      !is.na(P_value) & P_value < p_threshold
    },
    direction = case_when(
      is.na(HR) ~ NA_character_,
      HR > 1 ~ "Positive",
      HR < 1 ~ "Negative",
      TRUE ~ "Neutral"
    ),
    exposure_norm = normalize_token(Exposure),
    is_target = exposure_norm %in% names(highlight_lookup),
    target_name = unname(highlight_lookup[exposure_norm]),
    target_name = ifelse(is_target, target_name, NA_character_)
  )

if (all(is.na(df$HR))) {
  stop("Unable to parse HR from column 'Hazard ratio (95%CI)'.")
}

base_theme <- theme_classic(base_size = 20) +
  theme(
    text = element_text(color = "black"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(size = 36, color = "black", face = "bold"),
    axis.title.y = element_text(size = 48, color = "black", face = "bold", margin = margin(r = 10)),
    axis.text.x = element_text(size = 36, angle = 55, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 48),
    plot.title = element_text(size = 28, face = "bold"),
    plot.subtitle = element_text(size = 20),
    legend.title = element_text(size = 36),
    legend.text = element_text(size = 30),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )

# ---------------------------
# 1) Significant counts bar plot
# ---------------------------
message("Building significant counts bar plot...")

counts <- df %>%
  filter(!is.na(Outcome), Outcome != "NA", is_signif, direction %in% c("Positive", "Negative")) %>%
  count(Outcome, direction, name = "n") %>%
  complete(Outcome, direction = c("Positive", "Negative"), fill = list(n = 0)) %>%
  pivot_wider(names_from = direction, values_from = n, values_fill = 0) %>%
  mutate(total = Positive + Negative) %>%
  arrange(desc(total))

if (nrow(counts) == 0) {
  stop("No significant records found under current threshold settings.")
}

counts <- counts %>%
  mutate(Outcome = factor(Outcome, levels = Outcome))

counts_long <- counts %>%
  pivot_longer(cols = c("Positive", "Negative"), names_to = "direction", values_to = "count")

bar_plot <- ggplot(counts_long, aes(x = Outcome, y = count, fill = direction)) +
  geom_col(width = 0.78) +
  geom_text(
    data = counts,
    aes(x = Outcome, y = total, label = total),
    inherit.aes = FALSE,
    vjust = -0.35,
    size = 9,
    fontface = "bold"
  ) +
  scale_fill_manual(values = c("Positive" = "#1b9e77", "Negative" = "#d95f02")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(
    x = NULL,
    y = "No. significant associations",
    fill = ""
  ) +
  base_theme +
  theme(
    legend.position = "top",
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 34, color = "black", face = "bold", margin = margin(r = 10)),
    axis.text.y = element_text(size = 29)
  )

ggsave(barplot_file_pdf, bar_plot, device = "pdf", width = 24, height = 10)
message("Saved: ", barplot_file_pdf)

# ---------------------------
# 2) HR Manhattan plot by disease
# ---------------------------
message("Building HR Manhattan plot...")

df_manhattan <- df %>%
  filter(
    !is.na(P_value), P_value > 0,
    !is.na(logHR),
    !is.na(Outcome), Outcome != "NA",
    Outcome %in% diseases_to_plot
  ) %>%
  mutate(
    Outcome = factor(Outcome, levels = diseases_to_plot),
    color_group = ifelse(is_signif, as.character(Outcome), "Not Significant")
  )

present_diseases <- diseases_to_plot[diseases_to_plot %in% unique(as.character(df_manhattan$Outcome))]
if (length(present_diseases) == 0) {
  stop("No valid rows found for diseases_to_plot.")
}

# Keep disease colors consistent with protein_icd10_chart_largefont.py
python_disease_colors <- c(
  "G45" = "#1f77b4", "I05" = "#aec7e8", "I07" = "#ff7f0e", "I08" = "#ffbb78",
  "I10" = "#2ca02c", "I12" = "#98df8a", "I20" = "#d62728", "I21" = "#ff9896",
  "I25" = "#9467bd", "I26" = "#c5b0d5", "I27" = "#8c564b", "I31" = "#c49c94",
  "I33" = "#e377c2", "I34" = "#f7b6d2", "I35" = "#7f7f7f", "I36" = "#c7c7c7",
  "I37" = "#bcbd22", "I42" = "#dbdb8d", "I44" = "#17becf", "I45" = "#9edae5",
  "I46" = "#393b79", "I47" = "#5254a3", "I48" = "#6b6ecf", "I49" = "#9c9ede",
  "I50" = "#637939", "I51" = "#8ca252", "I63" = "#b5cf6b", "I65" = "#cedb9c",
  "I67" = "#8c6d31", "I69" = "#bd9e39", "I70" = "#e7ba52", "I71" = "#e7cb94",
  "I73" = "#843c39", "I74" = "#ad494a", "I77" = "#d6616b", "I78" = "#e7969c"
)

disease_colors <- python_disease_colors[present_diseases]
final_colors <- c(disease_colors, "Not Significant" = "grey85")

target_label_df <- df_manhattan %>%
  filter(is_target, is_signif)

legend_breaks <- present_diseases

hr_manhattan_plot <- ggplot(df_manhattan, aes(x = Outcome, y = logHR)) +
  geom_jitter(
    aes(color = color_group),
    width = 0.30,
    height = 0,
    alpha = 0.55,
    size = 2.0
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.7, color = "black") +
  scale_color_manual(values = final_colors, breaks = legend_breaks, name = "Disease") +
  scale_x_discrete(expand = expansion(mult = 0.01)) +
  scale_y_continuous(expand = expansion(mult = c(0.10, 0.14))) +
  geom_text_repel(
    data = target_label_df,
    aes(label = target_name),
    size = 4.0,
    color = "#b30000",
    fontface = "bold",
    max.overlaps = Inf,
    seed = 123,
    box.padding = 0.35,
    point.padding = 0.12,
    segment.alpha = 0.7,
    segment.color = "#b30000",
    min.segment.length = 0
  ) +
  labs(
    x = NULL,
    y = "log(Hazard Ratio)"
  ) +
  base_theme +
  theme(
    legend.position = "right",
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 34, color = "black", face = "bold", margin = margin(r = 10)),
    axis.text.x = element_text(size = 36, angle = 60, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 36)
  ) +
  guides(
    color = guide_legend(
      ncol = 2,
      keywidth = grid::unit(1.35, "lines"),
      keyheight = grid::unit(1.35, "lines"),
      override.aes = list(size = 5.5, alpha = 1)
    )
  )

ggsave(hr_plot_pdf, hr_manhattan_plot, device = "pdf", width = 24, height = 10)
message("Saved: ", hr_plot_pdf)

message("Done.")
