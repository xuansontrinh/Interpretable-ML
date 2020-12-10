# Load necessary libraries
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
library(data.table)
library(ranger)
library(xgboost)
library(mboost)
library(e1071)
library(mltools)
library(data.table)

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

load("../south-german-credit.Rda")
data[which(sapply(data, is.ordered))] <- sapply(data[which(sapply(data, is.ordered))], encode_ordinal)
cat_cols <- names(data)[which(sapply(data, is.factor))]
cat_cols <- cat_cols[cat_cols != "credit_risk"]
data <- data.table::as.data.table(data)
data <- one_hot(data,cols=cat_cols)
names(data) <- make.names(names(data))
int_cols <- names(data)[which(sapply(data, is.integer))]
data[, (int_cols) := lapply(.SD, as.numeric), .SDcols=int_cols]

task <- TaskClassif$new("german_credit", data, target = "credit_risk", positive = "good")
set.seed(11302020)
classif_err_measure = msr("classif.ce")
classif_xgboost_learner = lrn("classif.xgboost")
tuner = tnr("grid_search", resolution = 20)
tune_ps = ParamSet$new(list(
  ParamDbl$new("eta", lower = 0.05, upper = 1),
  ParamDbl$new("colsample_bylevel", lower = 0.25, upper = 1),
  ParamDbl$new("colsample_bytree", lower = 0.25, upper = 1),
  ParamInt$new("max_depth", lower = 2, upper = 20),
  ParamDbl$new("subsample", lower = 0.05, upper = 1),
  ParamInt$new("nrounds", lower = 5, upper = 100)
))
terminator = trm("evals", n_evals = 20)
at = AutoTuner$new(
  learner = classif_xgboost_learner,
  resampling = rsmp("holdout"),
  measure = classif_err_measure,
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)


design = benchmark_grid(
  tasks = task,
  learners = list(lrn("classif.log_reg"), lrn("classif.rpart"), at),
  resamplings = rsmp("cv", folds = 5L)
  # resamplings = rsmp("holdout", ratio=0.8)
)

bmr = benchmark(design)
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
