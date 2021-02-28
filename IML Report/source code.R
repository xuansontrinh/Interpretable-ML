library("mlr3viz")
library("ggplot2")
library("mlr3verse")
library("paradox")
theme_set(theme_bw())

set.seed(20211301)

load("south-german-credit.Rda")
task <- TaskClassif$new("german-credit",
  backend = data, target = "credit_risk", positive = "good"
)
task$col_roles$stratum <- "credit_risk"

fencoder <- po("encode",
  method = "one-hot",
  affect_columns = selector_type("factor")
)
ord_to_num <- po("colapply",
  applicator = as.numeric,
  affect_columns = selector_type("ordered"),
  id = "ord_to_num"
)

int_to_num <- po("colapply",
  applicator = as.numeric,
  affect_columns = selector_type("integer"),
  id = "int_to_num"
)

scaler <- po("scale",
  affect_columns = selector_type(c("integer", "numeric"))
)

po_over <- po("classbalancing",
  id = "oversample", adjust = "minor",
  reference = "minor", shuffle = FALSE, ratio = 2.3
)
pos <- int_to_num %>>% scaler %>>% ord_to_num %>>%
  fencoder %>>% po_over

inner_cv5 <- rsmp("cv", folds = 5L)
measure <- msr("classif.fbeta")
tuner <- tnr("random_search")
terminator <- trm("evals", n_evals = 40)

log_reg_learner <- po(lrn("classif.log_reg", predict_type = "prob"))
log_reg_pipeline <- pos %>>% log_reg_learner
log_reg_glearner <- GraphLearner$new(log_reg_pipeline,
  id = "log_reg", predict_type = "prob"
)

rpart_learner <- po(lrn("classif.rpart", predict_type = "prob"))
rpart_pipeline <- pos %>>% rpart_learner
rpart_glearner <- GraphLearner$new(rpart_pipeline,
  id = "rpart", predict_type = "prob"
)

fct_rpart_pipeline <- int_to_num %>>%
  scaler %>>% po_over %>>% rpart_learner
fct_rpart_glearner <- GraphLearner$new(fct_rpart_pipeline,
  id = "fct_rpart", predict_type = "prob"
)

ranger_pipeline <- pos %>>% po(lrn("classif.ranger",
  predict_type = "prob"
))
ranger_glearner <- GraphLearner$new(ranger_pipeline,
  id = "ranger", predict_type = "prob"
)

ranger_tune_ps <- ParamSet$new(list(
  ParamInt$new("classif.ranger.num.trees",
    lower = 100, upper = 140
  ), # number of trees
  ParamInt$new("classif.ranger.mtry",
    lower = 1,
    upper = ceiling(task$ncol / 2)
  ), # number of variables to possibly split at in each node
  ParamInt$new("classif.ranger.max.depth",
    lower = 2, upper = 20
  ) # maximum depth of the tree
))

ranger_at <- AutoTuner$new(
  learner = ranger_glearner,
  resampling = inner_cv5,
  measure = measure,
  search_space = ranger_tune_ps,
  terminator = terminator,
  tuner = tuner
)

fct_ranger_pipeline <- int_to_num %>>%
  scaler %>>% po_over %>>% po(lrn("classif.ranger", predict_type = "prob"))
fct_ranger_glearner <- GraphLearner$new(fct_ranger_pipeline,
  id = "fct_ranger", predict_type = "prob"
)

fct_ranger_at <- AutoTuner$new(
  learner = fct_ranger_glearner,
  resampling = inner_cv5,
  measure = measure,
  search_space = ranger_tune_ps,
  terminator = terminator,
  tuner = tuner
)

xgboost_learner <- po(lrn("classif.xgboost", predict_type = "prob"))
xgboost_pipeline <- pos %>>% xgboost_learner
xgboost_glearner <- GraphLearner$new(xgboost_pipeline,
  id = "xg_boost", predict_type = "prob"
)

xgboost_search_space <- ParamSet$new(list(
  ParamDbl$new("classif.xgboost.eta", lower = 0.1, upper = 0.3),
  ParamDbl$new("classif.xgboost.colsample_bytree", lower = 0.5, upper = 0.9),
  ParamInt$new("classif.xgboost.max_depth", lower = 8, upper = 16),
  ParamDbl$new("classif.xgboost.subsample", lower = 0.5, upper = 0.8),
  ParamInt$new("classif.xgboost.nrounds", lower = 110, upper = 118),
  ParamDbl$new("classif.xgboost.gamma", lower = 0, upper = 4),
  ParamDbl$new("classif.xgboost.lambda", lower = 1, upper = 4.5),
  ParamDbl$new("classif.xgboost.alpha", lower = 0, upper = 1)
))

# Creating the AutoTuner.
xgboost_at <- AutoTuner$new(
  learner = xgboost_glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = xgboost_search_space,
  tuner = tuner,
  measure = measure
)

linear_svm_learner <- po(lrn("classif.svm",
  type = "C-classification", kernel = "linear", predict_type = "prob"
))
poly_svm_learner <- po(lrn("classif.svm",
  type = "C-classification", kernel = "polynomial", predict_type = "prob"
))
radial_svm_learner <- po(lrn("classif.svm",
  type = "C-classification", kernel = "radial", predict_type = "prob"
))

# Pipelines
linear_svm_pipeline <- pos %>>% linear_svm_learner
poly_svm_pipeline <- pos %>>% poly_svm_learner
radial_svm_pipeline <- pos %>>% radial_svm_learner

# Learners
linear_svm_glearner <- GraphLearner$new(linear_svm_pipeline,
  id = "linear_svm", predict_type = "prob"
)
poly_svm_glearner <- GraphLearner$new(poly_svm_pipeline,
  id = "poly_svm", predict_type = "prob"
)
radial_svm_glearner <- GraphLearner$new(radial_svm_pipeline,
  id = "radial_svm", predict_type = "prob"
)

# Search spaces
poly_svm_search_space <- ParamSet$new(list(
  ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100),
  ParamDbl$new("classif.svm.gamma", lower = 0.0001, upper = 1),
  ParamInt$new("classif.svm.degree", lower = 1, upper = 4)
))

radial_svm_search_space <- ParamSet$new(list(
  ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100),
  ParamDbl$new("classif.svm.gamma", lower = 0.0001, upper = 1)
))

linear_svm_search_space <- ParamSet$new(list(
  ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100)
))

linear_svm_at <- AutoTuner$new(
  learner = linear_svm_glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = linear_svm_search_space,
  tuner = tuner,
  measure = measure
)

poly_svm_at <- AutoTuner$new(
  learner = poly_svm_glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = poly_svm_search_space,
  tuner = tuner,
  measure = measure
)

radial_svm_at <- AutoTuner$new(
  learner = radial_svm_glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = radial_svm_search_space,
  tuner = tuner,
  measure = measure
)

outer_cv3 <- rsmp("cv", folds = 3L)
design <- benchmark_grid(
  task = task,
  learners = list(
    log_reg_glearner,
    rpart_glearner,
    fct_rpart_glearner,
    ranger_at,
    fct_ranger_at,
    xgboost_at,
    linear_svm_at,
    poly_svm_at,
    radial_svm_at
  ),
  resamplings = outer_cv3
)
# benchmark
bmr <- benchmark(design)

bmr$aggregate(measure)
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = measure) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

autoplot(bmr, measure = msr("classif.fbeta")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = msr("classif.bacc")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = msr("classif.precision")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = msr("classif.recall")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = msr("classif.auc")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = msr("classif.bbrier")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
