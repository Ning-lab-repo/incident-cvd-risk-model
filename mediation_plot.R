#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(stringr)
  library(purrr)
})

# Keep the original workflow style: one circular plot per exposure
# Include TRIG, HBA1C, WC and BMI
EXPOSURE_PATHS <- c(
  "TRIG"  = "TableC_Mediation_Results_TRIG",
  "HBA1C" = "TableC_Mediation_Results_HBA1C",
  "WC"    = "TableC_Mediation_Results_WC",
  "BMI"   = "TableC_Mediation_Results_BMI"
)

# User-specified disease colors
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

DISEASE_CODES <- names(python_disease_colors)
DISEASE_FILES <- paste0("TableC_", DISEASE_CODES, "_Mediation.csv")

process_exposure <- function(exposure_name, exposure_path) {
  cat("Processing exposure:", exposure_name, "\n")
  all_disease_data <- list()
  
  for (disease_file in DISEASE_FILES) {
    file_path <- file.path(exposure_path, disease_file)
    if (!file.exists(file_path)) {
      cat("  Skipping missing file:", file_path, "\n")
      next
    }
    
    disease_data <- tryCatch(
      read_csv(file_path, col_types = cols(.default = "c")),
      error = function(e) {
        cat("  Error reading file:", file_path, "\n")
        cat("   ", e$message, "\n")
        NULL
      }
    )
    if (is.null(disease_data)) next
    
    disease_code <- str_extract(disease_file, "(?<=TableC_)[^_]+(?=_Mediation\\.csv)")
    disease_data$Exposure <- exposure_name
    disease_data$Disease <- disease_code
    all_disease_data[[disease_file]] <- disease_data
  }
  
  if (!length(all_disease_data)) {
    cat("  No files found for exposure:", exposure_name, "\n")
    return(NULL)
  }
  
  bind_rows(all_disease_data) %>%
    mutate(
      PM = as.numeric(PM),
      PM_p = as.numeric(PM_p)
    ) %>%
    filter(!is.na(protein), !is.na(PM), !is.na(PM_p))
}

all_exposures_list <- map2(names(EXPOSURE_PATHS), EXPOSURE_PATHS, process_exposure)
all_exposures_list <- all_exposures_list[!sapply(all_exposures_list, is.null)]
if (!length(all_exposures_list)) stop("No data read. Check paths/filenames.")

top_results_all <- all_exposures_list %>%
  bind_rows() %>%
  filter(PM_p < 0.05, PM > 0) %>%
  group_by(Exposure, Disease) %>%
  arrange(desc(PM), .by_group = TRUE) %>%
  slice_head(n = 3) %>%
  ungroup()

if (!nrow(top_results_all)) stop("No significant PM > 0 results; cannot plot.")

cat("\nTop results selected for plotting:\n")
print(top_results_all)

legend_text_size <- 8
legend_title_size <- 10

for (current_exposure in names(EXPOSURE_PATHS)) {
  cat("\nGenerating plot for Exposure:", current_exposure, "\n")
  
  plot_data <- top_results_all %>%
    filter(Exposure == current_exposure) %>%
    arrange(Disease, desc(PM))
  
  if (!nrow(plot_data)) {
    cat("  No significant results for", current_exposure, "Skipping.\n")
    next
  }
  
  plot_data <- plot_data %>%
    mutate(
      Disease = factor(Disease, levels = DISEASE_CODES),
      id = row_number(),
      PM_percent = PM * 100,
      PM_scaled = PM_percent * (2 / 3),
      angle = 90 - 360 * (id - 0.5) / n(),
      hjust_protein = if_else(angle < -90, 1, 0),
      angle_protein = if_else(angle < -90, angle + 180, angle),
      protein_y = -5
    ) %>%
    mutate(angle_protein = angle_protein + 180)
  
  n_bars <- nrow(plot_data)
  y_min <- -50
  y_max <- max(plot_data$PM_scaled, na.rm = TRUE) + 40
  label_size_pm <- if_else(n_bars > 50, 2.5, 4)
  label_size_protein <- if_else(n_bars > 50, 3, 4.5)
  
  pdf_filename <- paste0("mediation_circular_plot_", current_exposure, ".pdf")
  pdf(pdf_filename, width = 11.5, height = 11.5)
  
  p <- ggplot(plot_data, aes(x = as.factor(id), y = PM_scaled)) +
    geom_bar(
      aes(fill = Disease),
      stat = "identity",
      width = 1,
      color = "white",
      linewidth = 0.2
    ) +
    scale_fill_manual(
      values = python_disease_colors,
      breaks = DISEASE_CODES,
      drop = TRUE,
      na.translate = FALSE,
      guide = guide_legend(title = "Disease")
    ) +
    coord_polar(start = 0, clip = "off") +
    ylim(y_min, y_max) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      legend.position = "inside",
      legend.position.inside = c(0.86, 0.5),
      legend.justification = c(0, 0.5),
      legend.direction = "vertical",
      legend.margin = margin(0, 0, 0, 0, "pt"),
      legend.box.margin = margin(0, 0, 0, -40, "pt"),
      legend.text = element_text(size = legend_text_size, color = "black"),
      legend.title = element_text(size = legend_title_size, color = "black", face = "bold"),
      plot.margin = margin(0.3, 0.1, 0.3, 0.1, "cm"),
      plot.title = element_blank()
    ) +
    labs(title = NULL)
  
  p <- p + geom_text(
    aes(label = sprintf("%.1f%%", PM_percent), y = PM_scaled + 2),
    size = label_size_pm, color = "black"
  )
  
  p <- p + geom_text(
    aes(label = protein, y = protein_y, angle = angle_protein, hjust = hjust_protein),
    size = label_size_protein, color = "black"
  )
  
  print(p)
  dev.off()
  cat("  Saved:", pdf_filename, "\n")
}

cat("\nAll plots generated.\n")
