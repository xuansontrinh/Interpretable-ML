library("iml")
library("mlr3verse")
library("skimr")

set.seed(20211301)

# load dataset

task <- tsk("german_credit")

# 1 target, 20 features
# 18 factor features of which 3 are ordinal
# 3 numeric features (age, amount duration)
skimr::skim(task$data())


fencoder <- po("encode",
               method = "one-hot",
               affect_columns = selector_type("factor"))
ord_to_int <- po("colapply",
                 applicator = as.integer,
                 affect_columns = selector_type("ordered"))

# PipeOps
po_over <- po(
  "classbalancing",
  id = "oversample",
  adjust = "minor",
  reference = "minor",
  shuffle = FALSE,
  ratio = 2.3
)
pos <- po("scale") %>>%
  fencoder %>>% ord_to_int %>>% po_over

inner_cv5 <- rsmp("cv", folds = 5L)
measure <- msr("classif.bacc")
tuner <- tnr("grid_search", resolution = 7L)
terminator <- trm("evals", n_evals = 20)

radial_svm_learner <- lrn(
  "classif.svm",
  type = "C-classification",
  kernel = "radial",
  predict_type = "prob"
)
radial_svm_pipeline <-
  pos %>>% radial_svm_learner %>>% po("threshold")
radial_svm_glearner <-
  GraphLearner$new(radial_svm_pipeline, id = "radial_svm")
radial_svm_search_space <- ParamSet$new(list(
  ParamDbl$new("threshold.thresholds", lower = 0, upper = 1),
  ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100),
  ParamDbl$new("classif.svm.gamma", lower = 0.0001, upper = 1)
))
radial_svm_at <- AutoTuner$new(
  learner = radial_svm_glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = radial_svm_search_space,
  tuner = tuner,
  measure = measure
)

radial_svm_at$predict_type <- "prob"
radial_svm_at$train(task)

# hyperparameters
# radial_svm_at$model$tuning_instance$result
#    threshold.thresholds classif.svm.cost
# 1:            0.6666667              100
#    classif.svm.gamma learner_param_vals  x_domain
# 1:           0.16675         <list[14]> <list[3]>
#    classif.bacc
# 1:    0.6484142
radial_svm_at$model

data <- task$data()

x <- data[,-1]

model <-
  Predictor$new(radial_svm_at, data = x, y = data$credit_risk)

## Feature Effects

effect <-
  FeatureEffects$new(model, features = c("property"), method = "ale")
plot(effect, features = c("property"))

victor <-
  data.frame(
    "age" = 24,
    "amount" = 40000,
    "credit_history" = "no credits taken/all credits paid back duly",
    "duration" = 60,
    "employment_duration" = "< 1 yr",
    "foreign_worker" = "yes",
    "housing" = "rent",
    "installment_rate" = "<20",
    "job" = "skilled employee/official",
    "number_credits" = "1",
    "other_debtors" = "none",
    "other_installment_plans" = "none",
    "people_liable" = "0 to 2",
    "personal_status_sex" = "female : non-single or male : single",
    "present_residence" = "< 1 yr",
    "property" = "unknown / no property",
    "purpose" = "car (new)",
    "savings" = "... >= 1000 DM",
    "status" = "... >= 200 DM / salary for at least 1 year",
    "telephone" = "yes"
  )
