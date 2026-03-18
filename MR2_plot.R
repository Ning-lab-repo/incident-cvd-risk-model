#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(scales)
})

s13 <- read_csv("S13_ALL_protein_to_disease.csv", show_col_types = FALSE) %>%
  mutate(Direction = "Protein→Disease")
s14 <- read_csv("S14_ALL_disease_to_protein.csv", show_col_types = FALSE) %>%
  mutate(Direction = "Disease→Protein")

top_n   <- 50
x_range <- c(0.10, 30)

dat <- bind_rows(s13, s14) %>%
  mutate(
    Method  = if_else(Method == "Inverse variance weighted", "IVW", Method),
    OR      = exp(Beta),
    OR_low  = exp(Beta - 1.96 * SE),
    OR_high = exp(Beta + 1.96 * SE),
    log2OR      = log2(OR),
    log2OR_low  = log2(OR_low),
    log2OR_high = log2(OR_high),
   # FDR     = if ("P value" %in% names(.)) P value else p.adjust(`P value`, "BH"),
    Sig     = if_else(`P value` < 0.05, "P < 0.05", "N.S.")
  ) %>%
  filter(Method == "IVW") %>%
  group_by(Direction) %>%
  arrange(desc(abs(Beta))) %>%
  slice_head(n = top_n) %>%
  ungroup() %>%
  mutate(Pair = reorder(paste(Exposure, Outcome, sep = " → "), OR))

plot_one_direction <- function(direction_label, file_name, color_line) {
  df <- dat %>% filter(Direction == direction_label)
  if (nrow(df) == 0) {
    warning(sprintf("方向 %s 没有数据，跳过绘图。", direction_label))
    return()
  }
  
  p <- ggplot(df, aes(x = log2OR, y = Pair)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#7f7f7f") +
    geom_errorbarh(aes(xmin = log2OR_low, xmax = log2OR_high),
                   height = 0.25, linewidth = 0.7, color = color_line) +
    geom_point(aes(fill = Sig), shape = 21, size = 3, stroke = 0.8,
               color = color_line) +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) +
    scale_fill_manual(values = c("P < 0.05" = "white", "N.S." = "black"),
                      name = NULL) +
    labs(x = "log2(Odds ratio)", y = NULL) +
    theme_minimal(base_size = 16) +
    theme(
      text = element_text(family = "sans", color = "black"),
      axis.text.y = element_text(color = "black"),
      axis.text.x = element_text(color = "black"),
      axis.line.x = element_line(color = "black", linewidth = 0.6),
      axis.line.y = element_line(color = "black", linewidth = 0.6),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "#e5e5e5"),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      legend.position = c(0.98, 0.03),
      legend.justification = c(1, 0),
      legend.direction = "vertical",
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "transparent", color = NA),
      legend.margin = margin(0, 0, 0, 0),
      legend.box = "vertical"
    )
  
  pdf(file_name, width = 7, height = 9, family = "sans")
  print(p)
  dev.off()
  message("已生成 ", file_name)
}

plot_one_direction("Protein→Disease", "MR_protein_to_disease_forest.pdf", "#1f78b4")
plot_one_direction("Disease→Protein", "MR_disease_to_protein_forest.pdf", "#e66101")







#################################################################################################用fdr画图



#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(scales)
})

s13 <- read_csv("S13_ALL_protein_to_disease.csv", show_col_types = FALSE) %>%
  mutate(Direction = "Protein→Disease")
s14 <- read_csv("S14_ALL_disease_to_protein.csv", show_col_types = FALSE) %>%
  mutate(Direction = "Disease→Protein")

top_n   <- 50
x_range <- c(0.10, 10)

dat <- bind_rows(s13, s14) %>%
  mutate(
    Method  = if_else(Method == "Inverse variance weighted", "IVW", Method),
    OR      = exp(Beta),
    OR_low  = exp(Beta - 1.96 * SE),
    OR_high = exp(Beta + 1.96 * SE),
    log2OR      = log2(OR),
    log2OR_low  = log2(OR_low),
    log2OR_high = log2(OR_high),
    FDR     = if ("FDR_BH" %in% names(.)) FDR_BH else p.adjust(`P value`, "BH"),
    Sig     = if_else(FDR < 0.05, "FDR < 0.05", "N.S.")
  ) %>%
  filter(Method == "IVW") %>%
  group_by(Direction) %>%
  arrange(desc(abs(Beta))) %>%
  slice_head(n = top_n) %>%
  ungroup() %>%
  mutate(Pair = reorder(paste(Exposure, Outcome, sep = " → "), OR))

plot_one_direction <- function(direction_label, file_name, color_line) {
  df <- dat %>% filter(Direction == direction_label)
  if (nrow(df) == 0) {
    warning(sprintf("方向 %s 没有数据，跳过绘图。", direction_label))
    return()
  }
  
  p <- ggplot(df, aes(x = log2OR, y = Pair)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#7f7f7f") +
    geom_errorbarh(aes(xmin = log2OR_low, xmax = log2OR_high),
                   height = 0.25, linewidth = 0.7, color = color_line) +
    geom_point(aes(fill = Sig), shape = 21, size = 3, stroke = 0.8,
               color = color_line) +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) +
    scale_fill_manual(values = c("FDR < 0.05" = "white", "N.S." = "black"),
                      name = NULL) +
    labs(x = "log2(Odds ratio)", y = NULL) +
    theme_minimal(base_size = 16) +
    theme(
      text = element_text(family = "sans", color = "black"),
      axis.text.y = element_text(color = "black"),
      axis.text.x = element_text(color = "black"),
      axis.line.x = element_line(color = "black", linewidth = 0.6),
      axis.line.y = element_line(color = "black", linewidth = 0.6),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_line(color = "#e5e5e5"),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      legend.position = c(0.98, 0.03),
      legend.justification = c(1, 0),
      legend.direction = "vertical",
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_blank(),
      legend.key = element_rect(fill = "transparent", color = NA),
      legend.margin = margin(0, 0, 0, 0),
      legend.box = "vertical"
    )
  
  pdf(file_name, width = 7, height = 9, family = "sans")
  print(p)
  dev.off()
  message("已生成 ", file_name)
}

plot_one_direction("Protein→Disease", "MR_FDRprotein_to_disease_forest.pdf", "#1f78b4")
plot_one_direction("Disease→Protein", "MR_FDRdisease_to_protein_forest.pdf", "#e66101")
