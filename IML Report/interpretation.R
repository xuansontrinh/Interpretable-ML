library("iml")
library("mlr3viz")
library("ggplot2")
library("mlr3verse")
set.seed(20211301)



load("south-german-credit.Rda")
task <- TaskClassif$new("german-credit", backend = data, target = "credit_risk", positive = "good")


fencoder <- po("encode",
               method = "one-hot",
               affect_columns = selector_type("factor")
)
ord_to_int <- po("colapply",
                 applicator = as.integer,
                 affect_columns = selector_type("ordered")
)

# PipeOps
# filter_op <- po("filter", flt("mim"),
#   filter.nfeat = 3
# ) # feature filtering on mutual information maximization
po_over <- po("classbalancing",
              id = "oversample", adjust = "minor",
              reference = "minor", shuffle = FALSE, ratio = 2.3
)
pos <- po("scale") %>>%
  fencoder %>>% ord_to_int %>>% po_over

inner_cv5 <- rsmp("cv", folds = 5L)
measure <- msr("classif.bacc")
# measure <- msr("classif.fbeta")
tuner <- tnr("grid_search", resolution = 7L)
# tuner <- tnr("random_search")
terminator <- trm("evals", n_evals = 20)

# Radial SVM
radial_svm_learner <- lrn("classif.svm",
                          type = "C-classification", kernel = "radial", predict_type = "prob"
)
radial_svm_pipeline <- pos %>>% radial_svm_learner %>>% po("threshold")
radial_svm_glearner <- GraphLearner$new(radial_svm_pipeline, id = "radial_svm")
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

radial_svm_at$predict_type = "prob"

radial_svm_at$train(task)

radial_svm_at$model


german <- as.data.frame(task$data())
x = german[which(names(german) != "credit_risk")]
# x$age = as.integer(x$age)
model = Predictor$new(radial_svm_at, data = x, y = german$credit_risk)

# effect = FeatureEffects$new(model)
# plot(effect, features = c("employment_duration"))
eff <- FeatureEffect$new(model, feature = c("employment_duration"))
eff$plot()

eff <- FeatureEffect$new(model, feature = c("employment_duration"), method="pdp")
eff$plot()

eff <- FeatureEffect$new(model, feature = c("employment_duration"), method="ice")
eff$plot()

eff$set.feature("installment_rate")
eff$plot()

# ggplot(data = data, mapping = aes(x = property, fill = credit_risk)) + geom_bar()


# Feature Importance
imp <- FeatureImp$new(model, loss="f1")

# H-statistic Interaction
ia <- Interaction$new(model, feature = "job")
