library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library("mlr3viz")
library(paradox)

task <- tsk("german_credit")

fencoder <- po("encode",
  method = "treatment", affect_columns = selector_type("factor")
)

ord_to_int <- po("colapply",
  applicator = as.integer,
  affect_columns = selector_type("ordered")
)

# construct Pipeline
encoded_data <- fencoder %>>% ord_to_int
task_over <- encoded_data$train(task)

# Class imbalace

table(task_over$colapply.output$truth())
#
# good  bad
#  700  300

# oversample minority class (relative to minority class)
po_over <- po("classbalancing",
  id = "oversample", adjust = "minor",
  reference = "minor", shuffle = FALSE, ratio = 2.3
)
table(po_over$train(task_over)$output$truth())
#
# good  bad
#  700  690

# Tuning

xgboost_pipe <- po_over %>>%
  lrn("classif.xgboost")
ranger_pipe <- po_over %>>%
  lrn("classif.ranger")

xgboost_learner <- GraphLearner$new(xgboost_pipe)
ranger_learner <- GraphLearner$new(ranger_pipe)
xgboost_learner$param_set$values$oversample.ratio=2
ranger_learner$param_set$values$oversample.ratio=2


# https://xgboost.readthedocs.io/en/latest/parameter.html
xgboost_sp <- ParamSet$new(
  list(
    ParamDbl$new("classif.xgboost.eta", lower = 0.1, upper = 0.3),
    ParamDbl$new("classif.xgboost.colsample_bytree", lower = 0.5, upper = 0.9),
    ParamInt$new("classif.xgboost.max_depth", lower = 8, upper = 16),
    ParamDbl$new("classif.xgboost.subsample", lower = 0.5, upper = 0.8),
    ParamInt$new("classif.xgboost.nrounds", lower = 100, upper = 108),
    ParamDbl$new("classif.xgboost.gamma", lower = 0, upper = 4),
    ParamDbl$new("classif.xgboost.lambda", lower = 1, upper = 4.5),
    ParamDbl$new("classif.xgboost.alpha", lower = 0, upper = 1)
  )
)

ranger_ps <- ParamSet$new(
  list(
    ParamInt$new("classif.ranger.num.trees", lower = 100, upper = 140),
    ParamInt$new("classif.ranger.mtry",
      lower = 1, upper = ceiling(task_over$colapply.output$ncol
        / 2)
    ),
    ParamInt$new("classif.ranger.max.depth",
      lower = 2, upper = 20
    )
  )
)


inner_cv5 <- rsmp("cv", folds = 5L)
measure <- msr("classif.fbeta")
terminator <- trm("evals", n_evals = 20)
# Terminate after 36 evaluations
tuner_grid <- tnr("grid_search", resolution = 9)


learns <- list(
  AutoTuner$new(
    learner = xgboost_learner,
    resampling = inner_cv5,
    measure = measure,
    search_space = xgboost_sp,
    terminator = terminator,
    tuner = tuner_grid
  ),
  AutoTuner$new(
    learner = ranger_learner,
    resampling = inner_cv5,
    measure = measure,
    search_space = ranger_ps,
    terminator = terminator,
    tuner = tuner_grid
  ),
  lrn("classif.log_reg"),
  lrn("classif.rpart")
)

design <- benchmark_grid(
  tasks = task_over,
  learners = learns,
  resamplings = rsmp("cv", folds = 5L)
)

set.seed(11302020)
bmr <- benchmark(design, store_models = TRUE)

autoplot(bmr, measure = measure)