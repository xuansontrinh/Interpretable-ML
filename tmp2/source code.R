library("mlr3")
library("mlr3learners")
library("mlr3filters")
library("mlr3pipelines")
library("mlr3tuning")
library("mlr3viz")
library("paradox")

task <- tsk("german_credit")

fencoder <- po("encode",
               method = "one-hot",
               affect_columns = selector_type("factor")
)
ord_to_int <- po("colapply",
                 applicator = as.integer,
                 affect_columns = selector_type("ordered")
)

# learner <- lrn("classif.svm", type = "C-classification", kernel = "polynomial")
learner <- lrn("classif.svm")
filter <- flt("mim")
filter_op <- po("filter", flt("mim"),
                filter.nfeat = 3
) # feature filtering on mutual information maximization
po_over <- po("classbalancing",
              id = "oversample", adjust = "minor",
              reference = "minor", shuffle = FALSE, ratio = 2.3
)
pipeline <- filter_op %>>% fencoder %>>% ord_to_int %>>% po("scale") %>>% po_over %>>% learner
glearner <- GraphLearner$new(pipeline)

# Paramset
svm_tune_ps <- ParamSet$new(list(
  ParamInt$new("mim.filter.nfeat",
               lower = 1, upper = length(task$feature_names)
  ),
  ParamDbl$new("classif.svm.cost", lower = 0.01, upper = 100),
  # ParamDbl$new("classif.svm.gamma", lower = 0.0001, upper = 1),
  ParamFct$new("classif.svm.kernel", c("linear")),
  ParamFct$new("classif.svm.type", c("C-classification"))
  # ParamInt$new("classif.svm.degree", lower = 1, upper = 4)
  # ParamDbl$new("oversample.ratio", lower = 1, upper = 10)
))

inner_cv5 <- rsmp("cv", folds = 5L)
measure <- msr("classif.bacc")
tuner <- tnr("grid_search", resolution = 5L)
terminator <- trm("evals", n_evals = 20L)
auto_tuner <- AutoTuner$new(
  learner = glearner,
  resampling = inner_cv5,
  terminator = terminator,
  search_space = svm_tune_ps,
  tuner = tuner,
  measure = measure
)

outer_cv3 <- rsmp("cv", folds = 3L)
rr <- resample(task = task, learner = auto_tuner, resampling = outer_cv3)
rr$aggregate(measure = measure)
autoplot(rr, measure = measure)
