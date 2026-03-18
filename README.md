# incident-cvd-risk-model
Risk prediction of incident cardiovascular disease using proteomics and UK Biobank data

cox.py： After excluding baseline CVD and building survival data, this script runs large-scale parallel protein-by-outcome Cox regressions and outputs HRs with corrected p-values.
  
model_selcet_protein.py：This script selects key proteins for each CVD outcome using candidate features and XGBoost importance, then exports summary tables for modeling.
 
model.py：This script trains 10-fold cross-validated XGBoost models for each outcome, reports AUC/multiple metrics, and saves final models.
 
mediation.py： This script orchestrates the full pipeline for WC/HBA1C/TRIG/BMI by chaining Table A linear regression, Table B Cox analysis, and Table C mediation analysis.
 
mediation_plot.R：This script filters significant positive mediation proportions from Table C and draws circular bar plots by exposure.
 
mri.R：This script performs parallel linear association analyses between proteins and CMR traits and exports BH-adjusted results.
 
MR2.R： Using FinnGen and protein GWAS data, this script runs bidirectional MR (protein→disease and disease→protein) and aggregates S13/S14 outputs.
 
MR2_plot.R： This script visualizes bidirectional MR results as IVW-based forest plots under both raw p-value and FDR criteria.
 
shap.py： This script loads the Delphi model, uses SHAP to explain token-level risk contributions, and includes aggregated SHAP visualization routines.
 
ppi.py： This script extracts the first-degree PPI subnetwork around ACTA2 and MMP12, then exports the network figure and node/edge tables.
 
Stacked chart.py：This script summarizes protein source outcomes and plots a stacked bar chart for proteins linked to multiple diseases.
