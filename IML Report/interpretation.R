library("mlr3viz")
library("ggplot2")
library("mlr3verse")
library(paradox)
library(iml)

set.seed(20211301)

load("./south-german-credit.Rda")
data <- lapply(data, function(x) if(is.integer(x)) as.numeric(x) else x)
data <- lapply(data, function(x) if(is.ordered(x)) factor(x, ordered = FALSE) else x)
data <- as.data.frame(data)
x <- data[which(names(data) != "credit_risk")]


# counterfactuals
bob <- x[1,]
temp <-
  data.frame(
    "age" = 30,
    "amount" = 18424,
    "credit_history" = "no credits taken/all credits paid back duly",
    "duration" = 60, # 5 years
    "employment_duration" = "4 <= ... < 7 yrs",
    "foreign_worker" = "no",
    "housing" = "rent",
    "installment_rate" = "< 20",
    "job" = "manager/self-empl./highly qualif. employee",
    "number_credits" = "1",
    "other_debtors" = "none",
    "other_installment_plans" = "none",
    "people_liable" = "0 to 2",
    "personal_status_sex" = "male : married/widowed",
    "present_residence" = "1 <= ... < 4 yrs",
    "property" = "car or other",
    "purpose" = "business",
    "savings" = "... >= 1000 DM",
    "status" = "... >= 200 DM / salary for at least 1 year",
    "telephone" = "yes (under customer name)"
  )
bob[1,] = temp[,colnames(x)]

james <- x[1,]
temp <-
  data.frame(
    "age" = 30,
    "amount" = 6480, # average car price in the 70s
    "credit_history" = "no credits taken/all credits paid back duly",
    "duration" = 12,
    "employment_duration" = "4 <= ... < 7 yrs",
    "foreign_worker" = "no",
    "housing" = "rent",
    "installment_rate" = "< 20",
    "job" = "skilled employee/official",
    "number_credits" = "2-3",
    "other_debtors" = "none",
    "other_installment_plans" = "bank",
    "people_liable" = "0 to 2",
    "personal_status_sex" = "female : non-single or male : single",
    "present_residence" = "< 1 yr",
    "property" = "unknown / no property",
    "purpose" = "car (new)",
    "savings" = "unknown/no savings account",
    "status" = "... >= 200 DM / salary for at least 1 year",
    "telephone" = "no"
  )
james[1,] = temp[,colnames(x)]

task <- TaskClassif$new("german-credit",
                        backend = data, target = "credit_risk", positive = "good"
)

# TRAINING PART

# fencoder <- po("encode",
#                method = "one-hot",
#                affect_columns = selector_type("factor")
# )
# ord_to_num <- po("colapply",
#                  applicator = as.numeric,
#                  affect_columns = selector_type(c("ordered","integer"))
# )
# 
# int_to_num <- po("colapply",
#                  applicator = as.numeric,
#                  affect_columns = selector_type("integer")
# )
# 
# # PipeOps
# po_over <- po("classbalancing",
#               id = "oversample", adjust = "minor",
#               reference = "minor", shuffle = FALSE, ratio = 2.3
# )
# pos <- po("scale") %>>%
#   fencoder %>>% ord_to_num %>>% po_over
# 
# inner_cv5 <- rsmp("cv", folds = 5L)
# measure <- msr("classif.bacc")
# tuner <- tnr("grid_search", resolution = 7L)
# terminator <- trm("evals", n_evals = 20)
# 
# # Radial SVM
# radial_svm_learner <- lrn("classif.svm",
#                           type = "C-classification", kernel = "radial", predict_type = "prob"
# )
# radial_svm_pipeline <- pos %>>% radial_svm_learner %>>% po("threshold")
# radial_svm_glearner <- GraphLearner$new(radial_svm_pipeline, id = "radial_svm")
# radial_svm_search_space <- ParamSet$new(list(
#   ParamDbl$new("threshold.thresholds", lower = 0, upper = 1),
#   ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100),
#   ParamDbl$new("classif.svm.gamma", lower = 0.0001, upper = 1)
# ))
# 
# radial_svm_at <- AutoTuner$new(
#   learner = radial_svm_glearner,
#   resampling = inner_cv5,
#   terminator = terminator,
#   search_space = radial_svm_search_space,
#   tuner = tuner,
#   measure = measure
# )
# 
# radial_svm_at$predict_type <- "prob"
# 
# radial_svm_at$train(task)
# 
# radial_svm_at$model
# 
# saveRDS(radial_svm_at, file="tuned_radial.rds")

# load the pretrained radial basis model
radial_svm_at <- readRDS("tuned_radial.rds")

# Create the predictor using iml library
model <- Predictor$new(radial_svm_at, data = x, y = data$credit_risk)

# Feature effect

eff <- FeatureEffect$new(model, feature = c("age"))
eff$plot()

eff <- FeatureEffect$new(model, feature = c("employment_duration"), method = "pdp")
eff$plot()

eff <- FeatureEffect$new(model, feature = c("employment_duration"), method = "ice")
eff$plot()


# Feature Importance
bacc = function(truth, response, sample_weights = NULL) {
  if (is.null(sample_weights)) {
    sample_weights = rep(1, length(truth))
  } else {
    assert_numeric(sample_weights, lower = 0, any.missing = FALSE)
  }
  
  label_weights = vapply(split(sample_weights, truth), sum, NA_real_)
  sample_weights = sample_weights / label_weights[truth]
  sample_weights[is.na(sample_weights)] = 0
  
  1 - sum((truth == response) * sample_weights) / sum(sample_weights)
}

f1 <- function(actual, predicted) {
  tp <- length(actual[actual == "good" & predicted == "good"])
  fp <- length(actual[actual == "bad" & predicted == "good"])
  fn <- length(actual[actual == "good" & predicted == "bad"])
  if (tp == 0) {
    return(1)
  } else {
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    return(1 - (2 * precision * recall / (precision + recall)))
  }
}

imp_fbeta <- FeatureImp$new(model, loss = f1)
feature_importance_fbeta<- plot(imp_fbeta) + scale_x_continuous("Feature importance (loss: 1 - F1)")
imp_bacc <- FeatureImp$new(model, loss = bacc)
feature_importance_bacc<- plot(imp_bacc) + scale_x_continuous("Feature importance (loss: 1 - bacc)")

# H-statistic Interaction
ia <- Interaction$new(model)
ia_job <- Interaction$new(model, feature = "job")

model$predict(bob)
model$predict(james)

# Shapley contribution
shapley_bob <- Shapley$new(model, x.interest = bob)
shapley_james <- Shapley$new(model, x.interest = james)

# Counterfactuals

devtools::load_all("../iml", export_all = FALSE)
devtools::load_all("../counterfactuals", export_all = FALSE)
# generated by irace in folder appendix_irace
best_params <- readRDS("../best_configs.rds")

cf <- Counterfactuals$new(
  predictor = model,
  x.interest = bob,
  lower = 0,
  epsilon = 0,
  target = c(0.5, 1),
  generations = list(
    mosmafs::mosmafsTermStagnationHV(10),
    mosmafs::mosmafsTermGenerations(200)
  ),
  mu = best_params$mu,
  p.mut = best_params$p.mut, 
  p.rec = best_params$p.rec,
  p.mut.gen = best_params$p.mut.gen,
  p.mut.use.orig = best_params$p.mut.use.orig,
  p.rec.gen = best_params$p.rec.gen, 
  initialization = "icecurve",
  p.rec.use.orig = best_params$p.rec.use.orig,
)

# retrieve the counterfactuals of Bob
cf_diff <- cf$results$counterfactuals.diff
# filter only the counterfactuals with prediction greater than 0.51
cf_result_diff <- cf_diff[cf_diff$pred.pred >= 0.51,]
cf_result_diff <- cf_result_diff[order(-cf_result_diff$pred.pred),]
# filter features that are not meaningful for the counterfactuals interpretation
cf_result_diff <- cf_result_diff[cf_result_diff$age == 0,]
cf_result_diff <- cf_result_diff[cf_result_diff$foreign_worker == 0,]
cf_result_diff <- subset(cf_result_diff, 
                         select = -c(
                           dist.target, 
                           dist.x.interest, 
                           dist.train, 
                           pred.NA,
                           credit_history,
                           age, 
                           foreign_worker,
                           employment_duration,
                           installment_rate,
                           personal_status_sex,
                           other_debtors,
                           property,
                           other_installment_plans,
                           housing,
                           number_credits,
                           telephone,
                           present_residence,
                           people_liable,
                           nr.changed,
                           status
                         ))

# write.table(cf_result_diff, file = "cf_result_diff.csv",
#             sep = "\t", row.names = F)

a <- cf$plot_parallel(features = c("duration", "amount"), plot.x.interest = FALSE)
a <- a + scale_x_discrete(expand = c(0.1, 0.1), labels = c("duration", "credit amount"))
a
b <- cf$plot_surface(features = c("duration", "amount"))
b
c <- cf$plot_hv()
c