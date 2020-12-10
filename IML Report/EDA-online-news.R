# Load necessary libraries
library(keras)
library(tensorflow)

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



# The data set of our choice is Online News Popularity Data Set @online-news.
data <- read.csv("datasets\\OnlineNewsPopularity\\OnlineNewsPopularity.csv", header = TRUE)


# Remove 2 non-predictive features + 1 uniformly distributed feature  out of the dataset
data <- subset(data, select=-c(url, timedelta, n_non_stop_words))
data <- cbind(data, log(data$shares))
names(data)[names(data) == "log(data$shares)"] <- "log_shares"

# Remove outliers
data <- subset(data, select=-c(shares))
  
# data=data[!data$n_non_stop_words==1042,]
# data=data[!data$n_non_stop_words==0,]
data=data[!data$n_non_stop_unique_tokens==650,]
# boxplot(data$n_non_stop_words, ylab="n_non_stop_words")
# boxplot(data$n_non_stop_unique_tokens, ylab="n_non_stop_unique_tokens")
summary(data)
# #Combining Plots for EDA for visual analysis
# par(mfrow=c(2,2))
# for(i in 2:length(data)){
#   hist(
#     data[,i],
#     xlab=names(data)[i],
#     main = paste("[" , i , "]","Histogram of", names(data)[i])
#   )
# }

# #Converting categorical values from numeric to factor - Weekdays
# for (i in 29:35){
#   data[,i] <- factor(data[,i])
#   
# }
# 
# #Converting categorical values from numeric to factor - News subjects
# for (i in 11:16){
#   data[,i] <- factor(data[,i])
# }

#check classes of data after transformation
sapply(data, class)

# #Checking importance of news subjects(categorical) on shares
# for (i in 11:16){
#   
#   boxplot(log(data$shares) ~ (data[,i]), xlab=names(data)[i] , ylab="shares")
# }
# 
# #Checking importance of weekdays on news shares
# for (i in 29:35){
#   
#   boxplot(log(data$shares) ~ (data[,i]), xlab=names(data)[i] , ylab="shares")
# }


# Regression Task defining
task_reg_share = TaskRegr$new(id = "reg_share", backend = data, target = "log_shares")
task_classif_share = TaskClassif$new(id = "classif_share", backend = data, target = "log_shares")

# Set seed
set.seed(11302020)

# # Train-test split
# train_set = sample(task_reg_share$nrow, 0.8 * task_reg_share$nrow)
# test_set = setdiff(seq_len(task_reg_share$nrow), train_set)
# 
# lm_learner$train(task_reg_share, row_ids = train_set)
# prediction = lm_learner$predict(task_reg_share, row_ids = test_set)
# autoplot(prediction)
rmse_measure = msr("regr.rmse")
rsq_measure = msr("regr.rsq")
classif_err_measure = msr("classif.ce")
# prediction$score(rsq_measure)
# 
# resampling = rsmp("cv", folds = 5L)
# resampling$instantiate(task_reg_share)
# resampling$iters
# 
# rr = resample(task_reg_share,lm_learner,resampling, store_models = TRUE)
# print(rr)
# rr$aggregate(rsq_measure)
# rr$aggregate(rmse_measure)

# # Define neural network
# # Define a model
# 
# model = keras_model_sequential() %>%
#   layer_dense(units = 12L, input_shape = 57L, activation = "relu") %>%
#   layer_dense(units = 12L, activation = "relu") %>%
#   layer_dense(units = 1L, activation = "linear") %>%
#   compile(optimizer = optimizer_sgd(),
#           loss = "mean_squared_error",
#           metrics = "mean_squared_error")
# # Create the learner
# nnlearner = lrn("regr.keras")
# nnlearner$param_set$values$model = model
# nnlearner$param_set$values$batch_size = 1L
# radial_svm = lrn("regr.svm", kernel = "radial")

# # Feature selection
# filter = flt("information_gain")
# filter$calculate(task_reg_share)
# as.data.table(filter$scores)

# # Define blackbox model + fine tune
# xgboost_learner = lrn("regr.xgboost")
# tuner = tnr("grid_search", resolution = 20)
# tune_ps = ParamSet$new(list(
#   ParamDbl$new("eta", lower = 0.05, upper = 1),
#   ParamDbl$new("colsample_bylevel", lower = 0.25, upper = 1),
#   ParamDbl$new("colsample_bytree", lower = 0.25, upper = 1),
#   ParamInt$new("max_depth", lower = 2, upper = 10),
#   ParamDbl$new("subsample", lower = 0.05, upper = 1)
# ))
# terminator = trm("perf_reached", level=0.5)
# at = AutoTuner$new(
#   learner = xgboost_learner,
#   resampling = rsmp("holdout"),
#   measure = rsquare_measure,
#   search_space = tune_ps,
#   terminator = terminator,
#   tuner = tuner
# )

# Define blackbox model + fine tune
classif_xgboost_learner = lrn("classif.xgboost")
regr_xgboost_learner = lrn("regr.xgboost")
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
  learner = regr_xgboost_learner,
  resampling = rsmp("holdout"),
  measure = rsq_measure,
  search_space = tune_ps,
  terminator = terminator,
  tuner = tuner
)


# Baseline Models for Benchmarking
design = benchmark_grid(
  tasks = task_reg_share,
  learners = list(lrn("regr.lm"), lrn("regr.rpart"), at),
  resamplings = rsmp("cv", folds = 5L)
  # resamplings = rsmp("holdout", ratio=0.8)
)
bmr = benchmark(design)
autoplot(bmr) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
autoplot(bmr, measure = rsq_measure) + theme(axis.text.x = element_text(angle = 45, hjust = 1))

train_set = sample(task_reg_share$nrow, 0.8 * task_reg_share$nrow)
test_set = setdiff(seq_len(task_reg_share$nrow), train_set)

at$train(task_reg_share, row_ids = train_set)
pred = at$predict(task_reg_share, row_ids = test_set)
