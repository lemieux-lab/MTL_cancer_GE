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
