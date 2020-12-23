# Load necessary libraries
library(data.table)
library(mlr3)
library(mlr3viz)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3tuning)
library(mlr3keras)
library(mlr3filters)

library(paradox)

library(ggplot2)
library(GGally)
library(ranger)
library(xgboost)
library(mboost)
library(e1071)
library(mltools)
library(kernlab)

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

load("../south-german-credit.Rda")
original_data = data
data[which(sapply(data, is.ordered))] <- sapply(data[which(sapply(
  data,
  is.ordered
))], encode_ordinal)

cat_cols <- names(data)[which(sapply(data, is.factor))]
cat_cols <- cat_cols[cat_cols != "credit_risk"]
data <- data.table::as.data.table(data)
data <- one_hot(data, cols = cat_cols)
names(data) <- make.names(names(data))
int_cols <- names(data)[which(sapply(data, is.integer))]
data[, (int_cols) := lapply(.SD, as.numeric), .SDcols = int_cols]

task <- TaskClassif$new("german_credit", data,
  target = "credit_risk", positive = "good"
)

original_task <- TaskClassif$new("german_credit", original_data,
                                      target = "credit_risk", positive = "good"
)

classif_err_measure <- msr("classif.ce")
balanced_acc_measure <- msr("classif.bacc", aggregator = median)
tuner <- tnr("grid_search", resolution = 9L)
xgboost_tune_ps <- ParamSet$new(list(
  ParamDbl$new("eta", lower = 0.1, upper = 0.3), # controls the learning rate
  ParamDbl$new("colsample_bytree", lower = 0.5, upper = 0.9), # fraction of features (variables) supplied to a tree
  ParamInt$new("max_depth", lower = 8, upper = 16), # maximum depth of the tree
  ParamDbl$new("subsample", lower = 0.5, upper = 0.8), # fraction of sample supplied to a tree
  ParamInt$new("nrounds", lower = 100, upper = 108), # maximum number of iterations
  ParamDbl$new("gamma", lower = 0, upper = 4), # regularization controller
  ParamDbl$new("lambda", lower = 1, upper = 4.5), # L2 regularization
  ParamDbl$new("alpha", lower = 0, upper = 1) # L1 regularization
))
ranger_tune_ps <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 140), # number of trees
  ParamInt$new("mtry", lower = 1, upper = ceiling(task$ncol / 2)), # number of variables to possibly split at in each node
  ParamInt$new("max.depth", lower = 2, upper = 20) # maximum depth of the tree
))

original_ranger_tune_ps <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 140), # number of trees
  ParamInt$new("mtry", lower = 1, upper = ceiling(original_task$ncol / 2)), # number of variables to possibly split at in each node
  ParamInt$new("max.depth", lower = 2, upper = 20) # maximum depth of the tree
))

svm_tune_ps <- ParamSet$new(list(
  ParamFct$new("type", levels = c("C-classification")),
  ParamFct$new("kernel", levels = c("radial", "sigmoid")),
  ParamDbl$new("cost", lower = 0.1, upper = 10), # cost of constraints violation
  ParamDbl$new("gamma", lower = 0.1, upper = 10) # parameter needed for all kernels except linear
))
poly_svm_tune_ps <- ParamSet$new(list(
  ParamFct$new("type", levels = c("C-classification")),
  ParamFct$new("kernel", levels = c("polynomial")),
  ParamInt$new("degree", lower = 1, upper = 4), # parameter needed for kernel of type polynomial
  ParamDbl$new("cost", lower = 0.1, upper = 10), # cost of constraints violation
  ParamDbl$new("gamma", lower = 0.1, upper = 10) # parameter needed for all kernels except linear
))
terminator <- trm("evals", n_evals = 20)
xgboost_at <- AutoTuner$new(
  learner = lrn("classif.xgboost"),
  resampling = rsmp("cv", folds = 5L),
  measure = balanced_acc_measure,
  search_space = xgboost_tune_ps,
  terminator = terminator,
  tuner = tuner
)
ranger_at <- AutoTuner$new(
  learner = lrn("classif.ranger"),
  resampling = rsmp("cv", folds = 5L),
  measure = balanced_acc_measure,
  search_space = ranger_tune_ps,
  terminator = terminator,
  tuner = tuner
)
original_ranger_at <- AutoTuner$new(
  learner = lrn("classif.ranger"),
  resampling = rsmp("cv", folds = 5L),
  measure = balanced_acc_measure,
  search_space = original_ranger_tune_ps,
  terminator = terminator,
  tuner = tuner
)
svm_at <- AutoTuner$new(
  learner = lrn("classif.svm"),
  resampling = rsmp("cv", folds = 5L),
  measure = balanced_acc_measure,
  search_space = svm_tune_ps,
  terminator = terminator,
  tuner = tuner
)

poly_svm_at <- AutoTuner$new(
  learner = lrn("classif.svm"),
  resampling = rsmp("cv", folds = 5L),
  measure = balanced_acc_measure,
  search_space = poly_svm_tune_ps,
  terminator = terminator,
  tuner = tuner
)

set.seed(11302020)
design <- benchmark_grid(
  tasks = task,
  learners = list(lrn("classif.log_reg"), lrn("classif.rpart"), ranger_at, svm_at, xgboost_at),
  resamplings = rsmp("cv", folds = 5L)
)

bmr <- benchmark(design)

# set.seed(11302020)
# original_design <- benchmark_grid(
#   tasks = original_task,
#   learners = original_ranger_at,
#   resamplings = rsmp("cv", folds = 5L)
# )
# original_bmr <- benchmark(original_design)


# autoplot(bmr$filter(learner_ids = c(
#   "classif.log_reg", "classif.rpart", "classif.ranger", "classif.svm", "classif.xgboost.tuned"
# ))) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))
# 
# autoplot(bmr$filter(learner_ids = c(
#   "classif.log_reg", "classif.rpart", "classif.ranger", "classif.svm", "classif.xgboost.tuned"
# )), measure=balanced_acc_measure) +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1))

autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

autoplot(bmr, measure=balanced_acc_measure) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# autoplot(original_bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
# 
# autoplot(original_bmr, measure=balanced_acc_measure) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
