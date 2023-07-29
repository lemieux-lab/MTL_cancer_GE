setwd("/u/sauves/MTL_cancer_GE/")
library(dplyr)
require(tidyr)
library(survival)
library(ggplot2)
library(ggfortify)
# install.packages("survminer")
library(survminer)


brca_pca = read.csv("RES/SURV/TCGA_brca_pca_25_case_ids.csv")
brca_surv = read.csv("RES/SURV/TCGA_BRCA_clinical_survival.csv")
names(brca_surv)
#sum(brca_surv$case_id == brca_pca$case_id)
brca_surv$PAM50.mRNA[is.na(brca_surv$PAM50.mRNA)] = "undetermined"
brca = left_join(brca_pca, select(brca_surv, case_id, survt, surve, PAM50.mRNA) )
test_ids = sample(brca$case_id, nrow(brca) * 0.25)
brca_test = brca %>% filter(case_id %in% test_ids)
brca_train = brca %>% filter(!(case_id %in% test_ids))

mdl_fit = coxph(Surv(survt, surve) ~  pc_1 + pc_2 + pc_3 + pc_4 + pc_5 + pc_6 + pc_7 + pc_8 + pc_9 + pc_10 + 
                  pc_11 + pc_12 + pc_12 + pc_13 + pc_14 + pc_15 + pc_16 + pc_17 + pc_18 + pc_19 + pc_20 + 
                  pc_21 + pc_22 + pc_23 + pc_24 + pc_25 + PAM50.mRNA, data = brca_train)
preds = predict(mdl_fit, type = "risk", newdata = brca_test)
hist(preds)
risk = rep("low", nrow(brca_test))
risk[preds > median(preds)] = "high"
brca_fit = left_join(data.frame(case_id = test_ids, risk = risk ), brca_surv)
km_fit = survfit(Surv(survt, surve) ~ risk ,data = brca_fit )
autoplot(km_fit)


# data
#lgn = read.csv("Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv")
lgn = read.csv("Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv")
lgn_clin = read.csv("Data/LEUCEGENE/lgn_pronostic_CF")
# process
lgn_clin$sampleID
names(lgn)[1] = "sampleID"
names(lgn)
lgn$sampleID

lgn_table = left_join(lgn, lgn_clin)
lgn_table %>% select(Overall_Survival_Status, Overall_Survival_Time_days)
test_ids = sample(lgn_table$sampleID, nrow(lgn_table) * 0.25)
lgn_test = lgn_table %>% filter(sampleID %in% test_ids)
lgn_train = lgn_table %>% filter(!(sampleID %in% test_ids))
# train
mdl_fit = coxph(Surv(Overall_Survival_Time_days, Overall_Survival_Status) ~  ENSG00000174059 +
ENSG00000104341  + ENSG00000128805 + ENSG00000095932 + ENSG00000277988 + 
ENSG00000138722 + ENSG00000088305 + ENSG00000120833 + ENSG00000105810 + ENSG00000238049 + 
ENSG00000088882 + ENSG00000130584 + ENSG00000113657 + ENSG00000205978 + ENSG00000261633 + 
ENSG00000196139 + ENSG00000134531, data = lgn_train)
summary(mdl_fit)
# test
preds = -as.numeric(predict(mdl_fit, type = "risk", newdata = lgn_test))
#preds = -as.numeric(preds)
metrics = concordance(Surv(Overall_Survival_Time_days, Overall_Survival_Status) ~ preds, 
data = lgn_test)
metrics
hist(preds)
risk = rep("low", nrow(lgn_test))
risk[preds > median(preds)] = "high"
lgn_fit = left_join(data.frame(sampleID= test_ids, risk = risk ), lgn_clin)
km_fit = survfit(Surv(Overall_Survival_Time_days, Overall_Survival_Status) ~ risk ,
data = lgn_fit )
autoplot(km_fit) 
# plot

