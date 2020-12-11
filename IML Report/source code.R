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

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

load("../south-german-credit.Rda")
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
set.seed(11302020)
classif_err_measure <- msr("classif.ce")
classif_xgboost_learner <- lrn("classif.xgboost")
tuner <- tnr("grid_search", resolution = 5L)
tune_ps <- ParamSet$new(list(
  ParamDbl$new("eta", lower = 0.01, upper = 0.3), # controls the learning rate
  ParamDbl$new("colsample_bytree", lower = 0.5, upper = 0.9), # fraction of features (variables) supplied to a tree
  ParamInt$new("max_depth", lower = 2, upper = 20), # maximum depth of the tree
  ParamDbl$new("subsample", lower = 0.5, upper = 0.8), # fraction of sample supplied to a tree
  ParamInt$new("nrounds", lower = 100, upper = 140), # maximum number of iterations
  ParamDbl$new("gamma", lower = 0, upper = 5), # regularization controller
  ParamDbl$new("lambda", lower = 1, upper = 4.5), # L2 regularization
  ParamDbl$new("alpha", lower = 0, upper = 1) # L1 regularization
))
terminator <- trm("evals", n_evals = 5L)
at <- AutoTuner$new(
  learner = classif_xgboost_learner,
  resampling = rsmp("holdout"),
  measure = classif_err_measure,
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)


design <- benchmark_grid(
  tasks = task,
  learners = list(lrn("classif.log_reg"), lrn("classif.rpart"), at),
  resamplings = rsmp("cv", folds = 5L)
)

bmr <- benchmark(design)
autoplot(bmr$filter(learner_ids = c(
  "classif.log_reg",
  "classif.rpart", "classif.xgboost.tuned"
))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
