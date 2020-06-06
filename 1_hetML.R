
##################################################################################################################
#  
#  Description: The following code performs the Heterogeneity Analysis of Hirshleifer, et. al (2016) using machine
#       learning methods. It replicates the Table 6 of the paper and correspondingly estimates the 
#       Best Linear Predictor (BLP), Sorted Group Average Treatment Effects (GATES) and Classification Analysis (CLAN).
#       It uses the self-written function called BLP_GATES_CLAN() based on the code by code by Hannah Bull at 
#`      https://ml-in-econ.appspot.com/lab3.html and the code by Chernozhukov et. al. (2018) at
#       https://github.com/demirermert/MLInference`. 
#
#  Author: Joshi Mridul (M2, APE), Ucidami Mafeni (M2, PPD) and Purushottam Mohanty (M2, APE)
#  Date Modified: 05/06/2020
#
##################################################################################################################



rm(list=ls(all=TRUE))

# Load Packages
if (!require(pacman)) install.packages('pacman', repos = 'https://cran.rstudio.com')
pacman::p_load(tidyverse, stats, readstata13, foreach, data.table, lmtest, sandwich, stargazer, xtable,
               glmnet, randomForest, devtools, knitr, SuperLearner, caret, xgboost, gridExtra)

# install_github('susanathey/causalTree')
library(causalTree)

# setting main directory 
paths = c("/Users/purushottam/Documents/pse_local/coursework/machine_learning/MachineLearningHW", 
          "C:/Users/Mridul Joshi/Google Drive/MachineLearningHW", 
          "C:/Users/user/Desktop/Masters/M2/Machine learning/Project/MachineLearningHW")
names(paths) = c("purushottam", "mridul", "user")
user.path <- paths[Sys.info()[['user']]]
setwd(user.path)

maindir <- paste0(getwd())
datadir <- paste0(maindir, "/1_Data")
outputdir <- paste0(maindir, "/3_Output")


# setting seed
set.seed(1224)

###############   Replication ################## 

# import dataset
rep_df <- as.data.table(read.dta13(paste0(datadir, "/TurkeyPublicUseData.dta")))

# Values for categorical treatment variable
rep_df <- rep_df[maxTreat == "Control", Treat_assign := 0][maxTreat == "Treat", Treat_assign := 1]
rep_df <- rep_df[, maxTreat := NULL][, maxTreat := Treat_assign]

# Standardized employment outcome (of variables in table) (as done in the paper)
# Note this is missing if employed variable is missing, but otherwise is average of whatever variables are not missing
for(x in rbind("FF_work4wks1", "FF_work20hrs", "FF_numhrswk", "FF_inc_mth", "FF_ihs_inc_mth", "occstatus", "FF_workSS", "FF_inc_frm")) {

  thismean <- rep_df[maxTreat == 0, mean(eval(parse(text = x)), na.rm = TRUE)]
  thissd <- rep_df[maxTreat == 0, sd(eval(parse(text = x)), na.rm = TRUE)]
  rep_df[, paste0("z1_", x) := ((eval(parse(text = x)) - thismean) / thissd)]

}

rep_df[, employedindex2 := rowMeans(rep_df[, .(z1_FF_work4wks1, z1_FF_work20hrs, z1_FF_numhrswk, z1_FF_inc_mth, z1_FF_ihs_inc_mth, z1_FF_workSS, z1_occstatus)])]
rep_df[is.na(FF_work4wks1), employedindex2 := NA]



# *************** Table 6 -Heterogeneity with respect to pre-specified individual characteristics ******************

finaltable <- NULL

for (indvar in rbind("expectedbenefit", "posths", "prevcourse", "childunder6", "empoweredtowork", "raven", "numerate", "workcentral", "tenacity", "longterm")) {
  
  rep_df[, paste0("maxTreat_", indvar) := maxTreat * eval(parse(text = indvar))]
  table <- data.frame(1:2)
  
  for (depvar in rbind("FF_work20hrs", "employedindex2", "employjan12", "formallyemployedAug2013")) {
    
    stratadum <- paste0("stratadum", 1:457)
    form <- as.formula(paste0(depvar, "~ maxTreat + ", paste0("maxTreat_", indvar, " + ", indvar, " + "), paste(stratadum, collapse = " + ")))
    
    reg <- lm(form, data = rep_df[eval_course == 1,])
    se <- coeftest(reg, vcov = vcovHC(reg, "HC1"))
    
    coeftable <- data.frame(c(round(se[[3, 1]], digits = 3), paste0("(", round(se[[3, 2]], digits = 3), ")")))
    names(coeftable) <- c(depvar)
    table <- cbind(table, coeftable)
    
  }
  
  table <- cbind(c(indvar, indvar), c("coef", "se"), table)
  names(table)[[1]] <- "indvar"
  names(table)[[2]] <- "type"
  
  finaltable <- rbind(finaltable, table)
  
}

# format Latex table
a <- xtable(
  finaltable %>%
    mutate(indvar = as.character(indvar)) %>%
    mutate(indvar = ifelse(type == "se", "", indvar)) %>%
    select(-X1.2, -type) %>%
    
    rename("Employed 20+ Hours" = FF_work20hrs) %>%
    rename("Aggregate Employment Index" = employedindex2) %>%
    rename("Ever Formal by Jan 12" = employjan12) %>%
    rename("Currently Formal Aug 13" = formallyemployedAug2013)
  )

align(a) <- "llcccc"

# export latex table
print(a, booktabs = TRUE, floating = FALSE, latex.environments = "center", include.rownames = FALSE,
      file = paste0(outputdir, "/table6.tex"))




#################################### Heterogeneity Analysis using ML ############################################


# Source self-written BLP, GATES and CLAN function
source(paste0(getwd(), "/git/MLeco/2_hetML.R"))

# preparing data for ML (keeping only outcome variables, treatment, and covariates)
dataML <- rep_df %>%
  select(maxTreat, expectedbenefit, posths, prevcourse, childunder6, empoweredtowork, 
         raven, numerate, workcentral, tenacity, longterm, FF_work20hrs, employedindex2, employjan12,
         formallyemployedAug2013, contains("stratadum")) %>%
  filter(complete.cases(.))

# assignment to treatment
X <- dataML[["maxTreat"]]
# covariates
W <- dataML %>% select(expectedbenefit, posths, prevcourse, childunder6, empoweredtowork, 
                       raven, numerate, workcentral, tenacity, longterm, contains("stratadum"))


# vector of independent variable
vec_indvar <- rbind("expectedbenefit", "posths", "prevcourse", "childunder6", "empoweredtowork", "raven", 
                    "numerate", "workcentral", "tenacity", "longterm")

BLP_final <- NULL
GATES_final <- NULL
CLAN_final <- NULL

# no. of runs
run = 100

# loop over all dependent variables
# provides estimates, confidence intervals and p-values for BLP, GATES and CLAN
for (depvar in rbind("FF_work20hrs", "employedindex2", "employjan12", "formallyemployedAug2013")) {
  
  Y <- dataML[[paste0(depvar)]]
  
  # estimate BLP, GATES and CLAN for each run using self-written BLP_GATES_CLAN function
  allres <- rerun(run, BLP_GATES_CLAN(gates(Y, W, X))) 
  
  
  
  ############################## Best Linear Predictor ############################## 
  
  BLP_output <- NULL
  
  for (n in 1:run) {
    
    BLP <- allres[[n]]["BLP"]
    BLP <- as.data.frame(BLP)
    
    BLP_output <- rbind(BLP_output, BLP)
    
  }
  
  colnames(BLP_output) <-  sub("BLP.", "", colnames(BLP_output))
  
  # summarize coefficients, confidence intervals for all runs by median
  BLP_output <- BLP_output %>%
    bind_rows() %>%
    group_by(depvar, coefficient) %>%
    summarize_all(median)
  
  BLP_final <- rbind(BLP_final, BLP_output)
  
  
  ########################### Sorted Group Average Treatment Effects (GATES) ############################## 
  
  GATES_output <- NULL
  
  for (n in c(1:run)) {
    
    GATES <- allres[[n]]["GATES"]
    GATES <- as.data.frame(GATES)
    
    GATES_output <- rbind(GATES_output, GATES)
    
  }
  
  colnames(GATES_output) <-  sub("GATES.", "", colnames(GATES_output))
  
  # summarize coefficients, confidence intervals for all runs by median
  GATES_output <- GATES_output %>%
    bind_rows() %>%
    group_by(depvar, coefficient) %>%
    summarize_all(median)
  
  GATES_final <- rbind(GATES_final, GATES_output)
  
  
  ########################### Classification Analysis  (CLAN) ###########################
  
  CLAN_output <- NULL
  
  for (n in c(1:run)) {
    
    CLAN <- allres[[n]]["CLAN"]
    CLAN <- as.data.frame(CLAN)
    
    CLAN_output <- rbind(CLAN_output, CLAN)
    
  }
  
  colnames(CLAN_output) <-  sub("CLAN.", "", colnames(CLAN_output))
  
  # summarize coefficients, confidence intervals and p-values for all runs by median
  CLAN_output <- CLAN_output %>%
    bind_rows() %>%
    group_by(depvar, indvar) %>%
    summarize_all(median)
  
  CLAN_final <- rbind(CLAN_final, CLAN_output)
  
}



########################### Latex Tables ###########################


# BLP Table
BLP_table <- xtable(
  BLP_final %>%
    mutate(estimates = round(estimates, 3)) %>%
    mutate(ci_lower_90 = as.character(round(ci_lower_90, 3))) %>%
    mutate(ci_upper_90 = as.character(round(ci_upper_90, 3))) %>%
    mutate(CI = paste0("(", ci_lower_90, ",", ci_upper_90, ")")) %>%
    select(depvar, coefficient, estimates, CI) %>%
    
    gather(key = key, value = val, estimates:CI) %>%
    spread(coefficient, val) %>%
    arrange(depvar, desc(key)) %>%
    ungroup() %>%
    mutate(depvar = ifelse(key == "CI", "", depvar)) %>%
    select(-key) %>%
    mutate(depvar = case_when(
      depvar == "employedindex2" ~ "Employed 20+ Hours",
      depvar == "employjan12" ~ "Ever Formal by Jan 12",
      depvar == "FF_work20hrs" ~ "Aggregate Employment Index",
      depvar == "formallyemployedAug2013" ~ "Currently Formal Aug 13"
    )) 
)
    
align(BLP_table) <- "llcc"

print(BLP_table, booktabs = TRUE, floating = FALSE, latex.environments = "center", include.rownames = FALSE,
      file = paste0(outputdir, "/BLP_table.tex"))


# GATES Table
GATES_table <- xtable(
  GATES_final %>%
    mutate(estimates = round(estimates, 3)) %>%
    mutate(ci_lower_90 = as.character(round(ci_lower_90, 3))) %>%
    mutate(ci_upper_90 = as.character(round(ci_upper_90, 3))) %>%
    mutate(CI = paste0("(", ci_lower_90, ",", ci_upper_90, ")")) %>%
    select(depvar, coefficient, estimates, CI) %>%
    
    gather(key = key, value = val, estimates:CI) %>%
    spread(depvar, val) %>%
    arrange(coefficient, desc(key)) %>%
    ungroup() %>%
    mutate(coefficient = ifelse(key == "CI", "", coefficient)) %>%
    select(-key) %>%
    rename("Employed 20+ Hours" = FF_work20hrs) %>%
    rename("Aggregate Employment Index" = employedindex2) %>%
    rename("Ever Formal by Jan 12" = employjan12) %>%
    rename("Currently Formal Aug 13" = formallyemployedAug2013)
)

align(GATES_table) <- "llcccc"

print(GATES_table, booktabs = TRUE, floating = FALSE, latex.environments = "center", include.rownames = FALSE,
      file = paste0(outputdir, "/GATES_table.tex"))



# CLAN Table
CLAN_table <- xtable(
  CLAN_final %>%
    mutate(coef = round(coef, 3)) %>%
    mutate(cfl = as.character(round(cfl, 3))) %>%
    mutate(cfu = as.character(round(cfu, 3))) %>%
    mutate(pval = as.character(round(pval, 3))) %>%
    mutate(CI = paste0("(", cfl, ",", cfu, ")")) %>%
    mutate(pval = paste0("[", pval, "]")) %>%
    select(-cfl, -cfu) %>%
    
    gather(key = key, value = val, coef:CI) %>%
    spread(depvar, val) %>%
    arrange(match(indvar, vec_indvar), match(key, c("coef", "CI", "pval"))) %>%
    ungroup() %>%
    mutate(indvar = ifelse(key == "coef", indvar, "")) %>%
    select(-key) %>%
    rename("Employed 20+ Hours" = FF_work20hrs) %>%
    rename("Aggregate Employment Index" = employedindex2) %>%
    rename("Ever Formal by Jan 12" = employjan12) %>%
    rename("Currently Formal Aug 13" = formallyemployedAug2013) %>%
    
    mutate(indvar = case_when(
      indvar == "expectedbenefit" ~ "Expected benefit from ISKUR training",
      indvar == "posths" ~ "Post-high school education",
      indvar == "prevcourse" ~ "Previously taken part in a training course",
      indvar == "childunder6" ~ "Has a child aged 6 or under",
      indvar == "empoweredtowork" ~ "Is the main decision-maker for whether they work",
      indvar == "raven" ~  "Raven test score",
      indvar == "numerate" ~  "Numerate",
      indvar == "workcentral" ~ "Work centrality",
      indvar == "tenacity" ~ "Tenacity",
      indvar == "longterm" ~ "Unemployed for above the median duration"
    ))
)


align(CLAN_table) <- "llcccc"

print(CLAN_table, booktabs = TRUE, floating = FALSE, latex.environments = "center", include.rownames = FALSE,
      file = paste0(outputdir, "/CLAN_table.tex"))



## Plots for GATES

for (plotvar in rbind("FF_work20hrs", "employedindex2", "employjan12", "formallyemployedAug2013")) {
  
  graphs_df <- GATES_final %>%
    left_join(BLP_final, by = "depvar") %>%
    filter(coefficient.y %in% "ATE") %>%
    filter(depvar %in% paste0(plotvar))
  
  plot <- ggplot(data = graphs_df , aes(x = coefficient.x)) +
    geom_point(aes(y = estimates.x, colour = "GATES"), size = 3) +
    geom_errorbar(aes(ymax = ci_upper_90.x, ymin = ci_lower_90.x, y = estimates.x, width = .3, colour = "90% CB (GATES)")) +
    geom_hline(aes(yintercept = estimates.y[1], colour = "ATE")) +
    geom_hline(aes(yintercept = ci_upper_90.y, colour = "90% CB (ATE)"), linetype = "dashed") +
    geom_hline(aes(yintercept = ci_lower_90.y, colour = "90% CB (ATE)"), linetype = "dashed") + 
    scale_x_discrete(labels = c("1", "2", "3", "4")) +
    theme_classic() +
    ylab("Treatment Effect") + xlab("Group by Quartile") +
    theme(plot.title = element_text(hjust = 0.5,size = 18, face = "bold"), axis.title=element_text(size=10), legend.text=element_text(size=10), 
          legend.key = element_rect(colour = NA, fill = NA), legend.key.size = unit(1, 'lines'), legend.title=element_blank(),legend.justification= "center", 
          legend.position = "bottom", legend.direction = "horizontal", legend.background = element_rect(size = 0.5, colour = "black", linetype = "solid"))  +
    scale_colour_manual(values = c("blue", "black", "green", "black"),
                        breaks=c('ATE',"90% CB (ATE)","GATES",'90% CB (GATES)'),
                        guide = guide_legend(override.aes = list(
                          linetype = c("dashed", "dashed"  ,"blank", "solid"),
                          shape = c(NA,NA, 16, NA)), ncol = 4, byrow=TRUE)) 
  
  ggsave(plot, filename = paste0(outputdir, "/", plotvar, ".pdf"))
  
}









