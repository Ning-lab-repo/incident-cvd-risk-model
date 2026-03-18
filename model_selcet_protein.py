
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import os

# --- 文件路径和常量 ---
file_path = "/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/pro53013_morbidity_delphi_no_baseline_cvd.csv"
DIAG_COL = "new_diagnosis_after_baseline"
# DEATH_ICD_COL = "newp_s_alldead" # 移除了对死亡列的依赖
base_path = '/home/data/heamei/nmrLR/pro_cox/cvd_morbidity/model/verify/test_selectes_pro/test'  # 输出路径
if not os.path.exists(base_path):
    os.makedirs(base_path)

# --- 候选蛋白列表 (示例) ---
# 注意：实际使用时必须替换为完整的999个蛋白列表
candidate_proteins = [ "CLSTN2","GALNT5","NPPB","CD93","ROBO2","SCARF2","NTproBNP","SDC1","TNFRSF10A","TIMP1","ST6GAL1","GDF15","CST1","ENDOU","TNFRSF13B","ABCA2","PYY",
"MYL1","PRRT3","RNF149","EDN1","SCN4B","GRP","RNASE10","HRC","CCER2","AHNAK","FSTL1","AHNAK2","PYDC1","SIGLEC8","ADAMTSL4","MYL3","HSDL2","ROBO4","SERPINF1","C7","SOX9",
"SEZ6","EFNB2","SCT","CLGN","MYLPF","HS6ST2","INHBB","CD300A","CD7","SLITRK1","EPHA4","UNC5D","APOA4","NXPH3","TGFBR1","VEGFB","KLRF1","SPINK2","POF1B","LRG1","NEDD9",
"PTGES2","NPHS1","BTLA","CEP350","TPRKB","HMMR","CYTL1","DCLRE1C","FGF9","ITGBL1","PRAME","CHGA","IL25","PRG2","SLURP1","CLINT1","VGF","MYH7B","PRAP1","PSAP","PDCL2",
"CKB","GPR158","SLC28A1","TMOD4","SBSN","PCNA","PPP1R14D","VWC2L","MDH1","THAP12","FGF7","TIGIT","MAPK13","RASGRF1","SLIRP","DCUN1D2","PDIA5","MORF4L2","COL3A1","ESPL1",
"CD248","OCLN","ASGR2","CDH23","GPD1","FNDC1","ENOPH1","SPRR3","ADAM9","PXDNL","TARS1","VSIG2","PTPRR","SNX2","PALM2","PCDHB15","MAMDC4","AHSA1","TGFB2","KIAA0319",
"ITIH5","PBXIP1","INSL5","ZNRF4","FAM171B","LUZP2","GABARAP","GAST","EDDM3B","DUSP13","C9orf40","BCL7B","IGFBPL1","TACSTD2","SORCS2","EGFL7","ANKRD54","FSHB","COQ7",
"TAFA5","CD38","CALB1","CHCHD10","LMNB2","KRT14","SCG2","SEZ6L","KLK6","AMBP","HLA-E","MOG","PTPRN2","NXPH1","SHISA5","ORM1","RNASE1","RNASE6","BSG","CHGB","WFDC2",
"FGF23","GPR37","EFNA4","ASGR1","CD302","IGFBP4","RTN4R","GFRA1","PGF","TNFRSF4","EDA2R","ADM","CGA","EFNA1","SPINK1","HYOU1","TNFRSF1A","GCHFR","ACY3","BHMT2","CD2",
"ATP6V1G2","HADH","KHK","ECHS1","TMEM132A","SEL1L","SEC31A","C3","SH3GL3","PENK","NHLRC3","IL31RA","FGF16","BRD1","PRUNE2","C1QL2","SMPD3","CD82","ESR1","TYRP1","CPOX",
"MRPL52","AKR7L","CDH22","IFIT3","EPN1","SHH","TEX33","GLI2","KIAA1549L","OGT","MTHFSD","GUCY2C","ITGAL","CA7","DNAJA1","GIPR","CCND2","HRAS","TLR2","SAT2","TXK",
"CREBZF","RNASE4","FGF3","TLR1","ITGAX","ACRV1","PKN3","CD36","SLC34A3","TK1","LACRT","MYH9","AMOTL2","IFI30","CLEC3B","PLG","CFB","TCN1","RAB44","SERPINA4","MBL2",
"BNIP2","UPB1","MELTF","ASPN","RBFOX3","ECHDC3","PTPRB","SRPX","SCRG1","PTGR1","FABP3","S100A14","SORBS1","RTN4IP1","RLN1","NPC2","PHACTR2","LCN15","PTPRZ1","SUOX",
"CACNB3","B2M","CFD","BMPER","ATRAID","CRELD1","GM2A","ADGRD1","MAMDC2","EFCAB14","YAP1","FGL1","SEPTIN8","IMMT","IL22","SERPIND1","FAM20A","ADAMTSL2","EPHA2",
"SFTPA2","CTSD","IL17C","NT5C1A","TNF","LILRB4","OSM","FGF2","EGLN1","BPIFB1","PIGR","NOS3","CHI3L1","TGFA","PPIB","CCL7","DCXR","MMP12","RETN","SCARB2","IL6",
"TNFRSF10C","MEGF9","FAP","MMP7","COL6A3","QPCT","ACE2","ACP5","PGLYRP1","LTBR","PLAUR","NEFL","RBP5","PRSS8","ACVRL1","IGSF3","ADGRG1","BAIAP2","REG1A","TNFRSF10B",
"TNFRSF9","LGALS9","TNFRSF6B","VWC2","HSPG2","TNFRSF11B","SPON1","LAIR1","FSTL3","IL2RA","COLEC12","CDCP1","CD274","EFEMP1","MZB1","PLA2G2A","REG4","PTGDS","LGALS4",
"TNFRSF1B","SPON2","RELT","AGRN","CKAP4","CCL21","CSF1","CD300E","TFF3","KLK4","TFF2","CHRDL1","NECTIN2","PIK3IP1","IL18BP","HAVCR2","REN","CD27","AREG","CXCL16",
"NPDC1","NBL1","CXCL13","CDH2","TNFRSF12A","FAM3C","MSR1","CST3","CXCL17","LTBP2","VSIG4","RARRES2","HGF","CD83","IL4R","LAMP3","CD163","CD79B","CPM","ITGA5","NOMO1",
"COL18A1","CD74","FGFR2","KRT19","SHMT1","HAVCR1","TPP1","PLA2G15","IGFBP7","CCL16","SIGLEC1","VWA1","B4GALT1","CXCL9","NFASC","CTSO","CD59","LY6D","DSC2",
"TNFRSF14","IL10RB","IL12RB1","ULBP2","LAYN","CDNF","DCBLD2","TGFBR2","TFPI2","CTSL","CD4","IGSF8","RSPO3","ANGPT2","EPHB4","FABP4","GGH","CA6","SPP1",
"DTNB","HMCN2","PCBD1","SFRP4","HEG1","COL5A1","CHAD","CTSE","SERPINI1","ST13","LMOD1","GMPR2","AMOT","SLC9A3R1","ARHGAP45","DAAM1","LZTFL1","MFAP4",
"SNCA","GNPDA2","TPD52L2","PRKG1","CEACAM6","CD80","CTHRC1","CFP","CILP","PROS1","GSN","LCAT","ACTA2","SLAMF8","BCAN","MSLN","LRRN1","SPINT1","ICAM1",
"ADA2","VCAM1","EPO","IL15","APOM","SLAMF7","SELPLG","THBS2","CLEC14A","CXCL8","NRP1","WARS","TNC","AOC3","COL4A1","SFTPD","KITLG","ICAM5","CRIM1","NCAN",
"DLL1","TXNDC15","BST2","EGFR","EBI3_IL27","NOTCH3","ADIPOQ","IGFBP2","PRND","GIPC2","FBLN2","ELN","MRC1","MMP1","CD14","CRIP2","MDK","PLXNB2","CX3CL1",
"PRCP","RNASET2","LILRA5","IGF2R","LAMA4","ITGA11","LAG3","PDCD1LG2","CD276","PILRA","CXCL10","TNFSF13","BTN2A1","FOLR1","FGFBP1","PDCD1","TNFRSF8","FUT3_FUT5",
"CD300C","CXCL11","TNFSF13B","TCN2","TNFRSF11A","CCL20","CCL15","REG1B","CCN3","REG3A","CRISP3","HSPB6","TNFRSF19","IL19","OSCAR","OGN","CFH","MERTK","APCS",
"HJV","CFI","C1S","NPL","IL15RA","RRM2","PHOSPHO1","CCN1","CCL22","IL1RN","CLUL1","TFPI","SIGLEC7","PLA2G7","LDLR","TREM2","CRHBP","INHBC","FURIN","LGMN","SMOC2",
"STC1","GOLM2","PLAT","PLTP","NUCB2","SMOC1","MYBPC1","B3GNT7","TNFRSF17","LRTM2","EFHD1","IGLC2","BCAT1","TNNI3","CEACAM5","CLEC6A","CCL14","AGR2","SPINK4","CCL3",
"KLK11","UMOD","MAD1L1","IL10","MILR1","MVK","WNT9A","NOS1","DPP4","ANGPTL4","PI3","CALCA","LY96","IL12A_IL12B","GRPEL1","RBP2","DEFA1_DEFA1B","IL18","BTN3A2","CLEC5A",
"SIGLEC10","CLEC4D","TFF1","CCL17","SIRPB1","CD40","FABP1","SEMA3F","MATN2","CEACAM1","IL1R1","SIAE","SMPD1","ENPP5","SPINT2","TIMP4","GAS6","ITIH3","ADAM8","DPY30","VEGFA",
"IL1RL1","TGFB1","SERPINA11","CNTN1","DLK1","IL12B","MXRA8","LPA","IGSF9","A1BG","MUC13","RAB6A","CD300LG","DSG2","CTSB","PON3","LEP","FGF21","CA14","IDUA","NTRK3",
"PCSK9","ACTN2","MYOM3","GHR","F10","SCPEP1","GLA","MENT","HPSE","HHEX","BPIFB2","LECT2","PGA4","PON1","LGALS3BP","IL1RL2","SLAMF1","ADAM15","PTN","RSPO1","ANGPTL1",
"RET","PTX3","SSC4D","CDHR5","F7","ESM1","PROC","BCAM","CNTN3","B4GAT1","SSC5D","OSMR","NELL2","DSG3","VEGFD","WIF1","IGFBP1","ENPP2","MB","ENG","LTBP3","GCG","PON2",
"PAG1","HNMT","DEFB4A_DEFB4B","CSTB","SIT1","CCL18","DDAH1","KRT18","VMO1","GALNT10","MME","ADAMTS15","POLR2F","HYAL1","P4HB","ALDH3A1","CCL27","ADGRG2","LPL","LGALS1",
"GGT1","HSD11B1","BCL2L11","CES1","CCL4","MFGE8","F9","CTSZ","CCL23","TIMP3","PDGFB","CCL5","EPS8L2","NCS1","NPY","SERPINE1","IL18R1","CCL11","KYNU","AGRP","CTSS","APOD",
"ASS1","GFRAL","KRT8","PDZK1","ADAM12","ERMAP","MTDH","PPL","PTP4A3","LATS1","DMP1","CBS","POMC","LAMB1","ENPP6","C1QTNF9","WFDC1","FUOM","RIDA","ADH1B","NAGPA","DIPK2B",
"MOCS2","CD72","KCTD5","MTUS1","SERPINH1","BCHE","TTR","KLK15","GSR","GBP1","HIP1R","NIT2","TALDO1","FTCD","SMTN","CHCHD6","GPRC5C","FNTA","VSNL1","MAP2","CLSTN3","DNPEP",
"PCYT2","SSNA1","ZNRD2","SLC9A3R2","TNFAIP2","LETM1","OTUD7B","FCN1","STAB2","IGDCC4","NPTX2","TREH","TCTN3","CPXM2","UBE2L6","ADAMTSL5","MYDGF","SERPING1","APOC1","CPB2",
"ACE","APOF","SHBG","PI16","GPI","AGT","C1RL","AFM","LAMP2","ENPP7","CDH1","PTS","LCN2","CRH","ACAA1","GSTA3","GFER","FCRLB","CD28","HPGDS","FCGR2B","CFC1","NDUFS6","BAX",
"C19orf12","PBLD","ANXA10","PTK7","NFATC3","FGF5","LAP3","ADH4","CA5A","CEACAM8","VSTM2L","IL17RB","SFRP1","SETMAR","IL13RA1","IFNGR2","SNCG","PPP3R1","GGT5","FKBP4","TNFSF14",
"PTPN1","FKBP5","GPC5","DNMBP","TXLNA","DPP7","CAPG","ERP44","AIFM1","ARG1","TYMP","CCL28","MCFD2","QDPR","THOP1","CDHR2","AMIGO2","CA12","FAM3B","ARSB","APBB1IP","ALPP","DTX3",
"CDKN1A","HS3ST3B1","ABL1","IQGAP2","LYN","OMG","SMAD5","SCAMP3","YES1","CANT1","GRN","TINAGL1","ERBB3","DBI","SULT2A1","CXADR","CLEC4G","LIFR","PTH1R","NPPC","IFNLR1","SLC39A5",
"FCAR","MAPK9","HEXIM1","BID","CD46","DNER","GSTA1","VSIR","SORT1","CTSC","IL1R2","CPE","AGER","ACY1","AGXT","FBP1","PILRB","APLP1","NADK","LRP11","CCDC80","MARCO","TSHB","PLIN3",
"IL6ST","MMP9","FIS1","ANG","PAMR1","NCAM2","SOD2","ARSA","CD99L2","IL1RAP","TXNRD1","F11R","ITGAM","TMSB10","SUSD2","WFIKKN2","NUDT5","TNXB","MMP3","CNTN5","ANGPTL2","CST7","EPHA1",
"HSPA1A","MET","CD48","F2R","CRELD2","ANGPTL3","SORD","LGALS3","NELL1","ERBB2","CTSF","KDR","ADAMTS8","FLT4","CD1C","HTRA2","ATOX1","NT5E","SNAP29","PPP1R12A","CPXM1","CHL1","CD34",
"OMD","DAG1","FABP5","ANPEP","CRLF1","KIT","GUSB","FETUB","C2","SCLY","HBEGF","CDH15","C1QTNF1","PRSS2","LBP","ART3","SELP","MMP8","OXT","PVR","LBR","RGMA","TXNDC5","LGALS8","SCARA5",
"EZR","CD5","CLEC1B","HAO1","ITGAV","CCN4","ANGPTL7","TRIAP1","FASLG","KLRB1","TNFSF12","TNFSF10","LAT","EIF4G1","KLRD1","CXCL6","IL16","SELE","PTPRS","CDH5","SCARF1","COL1A1","NOTCH1","CPB1","CPA1","ESAM"
]

# --- 结局列表 (36个) ---
outcomes = [
    "I10","I50","I25","I73","I48","I27","I12","I21","I70","I65","I77","I51","I46","I67","I63","I20","I35","I74",
  "I69","I71","I26","I08","I34","I44","I05","I47","I37","I45","I78","I07","I42","G45","I49","I31","I33","I36"
]

# --- 1. 加载数据 ---
print("开始加载数据...")
try:
    data = pd.read_csv(file_path)
    print(f"数据加载成功，总计 {len(data)} 行。")
except Exception as e:
    print(f"错误：无法加载文件 {file_path}。{e}")
    exit()

# --- 2. 提取并填充候选蛋白特征 ---
print("开始填充缺失值（中位数）...")
imputer = SimpleImputer(strategy='median')
X_all = data[candidate_proteins]
X_all_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=candidate_proteins, index=data.index)
# data[candidate_proteins] = X_all_imputed  # 更新原始data中的数据（如果后续需要使用data）
print("缺失值填充完成。")

# --- 3. 定义单个结局的处理函数 ---
def process_outcome(outcome):
    """
    处理单个结局：定义y，训练XGB，选择Top蛋白。
    返回: (outcome, event_count, top_proteins_list)
    """
    
    # 定义y：检查诊断(DIAG_COL)中是否有以outcome开头的代码（前三位匹配）
    # ** 已按要求移除 DEATH_ICD_COL **
    diag_list = data[DIAG_COL].fillna('').astype(str).str.split('|')
    y = diag_list.apply(lambda codes: any(code.startswith(outcome) for code in codes)).astype(int)
    
    event_count = sum(y)
    
    # 如果没有阳性样本，无法训练
    if event_count == 0:
        print(f"结局 {outcome} 没有阳性事件，跳过训练。")
        return (outcome, 0, [])
    
    # 训练XGB分类器（使用gain作为重要性）
    model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    min_child_weight=5,
   # gamma=0.5,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.01,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=len(y) / event_count,
    random_state=42,
    n_jobs=-1,
    importance_type='gain'
    )
    model.fit(X_all_imputed, y)
    
    # 获取重要性并归一化（和为1）
    importances = model.feature_importances_
    if np.sum(importances) == 0:
        print(f"结局 {outcome} 训练完成，但模型总重要性为0。")
        return (outcome, event_count, [])
        
    norm_imp = importances / np.sum(importances)
    
    # 按重要性降序排序，累加直到>=0.3
    sorted_idx = np.argsort(norm_imp)[::-1]
    cum_imp = np.cumsum(norm_imp[sorted_idx])
    
    # 如果总重要性<0.3，则全部选取 (这种情况不应该发生，但作为保险)
   # if np.max(cum_imp) < 0.5:
    #    top_proteins = [candidate_proteins[i] for i in sorted_idx]
  #  else:
        # 找到第一个累加>=0.3的位置
    idx = np.where(cum_imp >= 0.03)[0][0] + 1
    top_proteins = [candidate_proteins[sorted_idx[j]] for j in range(idx)]
    
    print(f"结局 {outcome} 处理完成。事件数: {event_count}。选出 {len(top_proteins)} 个蛋白。")
    return (outcome, event_count, top_proteins)

# --- 4. 并行处理所有结局 ---
print(f"开始并行处理 {len(outcomes)} 个结局...")
n_cores = os.cpu_count()
results = Parallel(n_jobs=min(n_cores, len(outcomes)))(
    delayed(process_outcome)(outcome) for outcome in outcomes
)
print("所有结局处理完毕。")

# --- 5. 收集和整理结果 ---
all_top_proteins_set = set()
protein_source_list = []  # 用于新表1: [ {'protein', 'source_outcome'} ]
event_counts_list = []    # 用于新表2: [ {'outcome', 'event_count'} ]

for outcome, event_count, top_list in results:
    # 收集事件计数（新表2）
    event_counts_list.append({'outcome': outcome, 'event_count': event_count})
    
    # 收集蛋白来源（新表1）
    for protein in top_list:
        protein_source_list.append({'selected_protein': protein, 'source_outcome': outcome})
    
    # 收集最终蛋白并集（原始表）
    all_top_proteins_set.update(top_list)

# --- 6. 输出和保存 ---

# 输出 1: 事件计数表 (新要求)
df_event_counts = pd.DataFrame(event_counts_list)
df_event_counts = df_event_counts.sort_values(by='event_count', ascending=False)
event_count_path = os.path.join(base_path, 'outcome_event_counts5.csv')
df_event_counts.to_csv(event_count_path, index=False, encoding='gbk') # 使用gbk编码保存
print(f"\n--- 表1：事件计数 (已保存至 {event_count_path}) ---")
print(df_event_counts.to_string()) # 使用to_string()确保打印完整

# 输出 2: 蛋白来源表 (新要求)
df_protein_source = pd.DataFrame(protein_source_list)
df_protein_source = df_protein_source.drop_duplicates().sort_values(by=['selected_protein', 'source_outcome'])
protein_source_path = os.path.join(base_path, 'selected_proteins_by_source5.csv')
df_protein_source.to_csv(protein_source_path, index=False, encoding='gbk')
print(f"\n--- 表2：按来源分的蛋白列表 (已保存至 {protein_source_path}) ---")
print(f"总计 {len(df_protein_source)} 条来源记录。")
print(df_protein_source.head()) # 打印前几行示例

# 输出 3: 最终蛋白特征集 (原始要求)
all_top_proteins = sorted(list(all_top_proteins_set))
final_list_path = os.path.join(base_path, 'selected_proteins_for_modeling5.csv')
df_final_proteins = pd.DataFrame({'selected_proteins': all_top_proteins})
df_final_proteins.to_csv(final_list_path, index=False, encoding='gbk')
print(f"\n--- 表3：最终用于建模的蛋白特征集 (已保存至 {final_list_path}) ---")
print(f"总计 {len(all_top_proteins)} 个独立蛋白。")
print(all_top_proteins)

print("\n--- 脚本执行完毕 ---")