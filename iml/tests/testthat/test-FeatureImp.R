context("FeatureImp()")

set.seed(42)

expected_colnames = c("feature", "importance.05", "importance",
		      "importance.95", "permutation.error")

test_that("FeatureImp works for single output", {
  var.imp = FeatureImp$new(predictor1, loss = "mse")
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_false("data.table" %in% class(dat))
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
  p = plot(var.imp, sort = FALSE)
  expect_s3_class(p, c("gg", "ggplot"))
  p
  
  p = var.imp$plot()
  expect_s3_class(p, c("gg", "ggplot"))
  p
})

test_that("FeatureImp works for single output with single repetition", {
  var.imp = FeatureImp$new(predictor1, loss = "mse", n.repetitions = 1)
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_false("data.table" %in% class(dat))
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))  
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
})

test_that("FeatureImp with difference", {
  var.imp = FeatureImp$new(predictor1, loss = "mse", compare = "difference")
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_false("data.table" %in% class(dat))
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))  
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
})

test_that("FeatureImp with 0 model error", {
  data(iris)
  require("mlr")
  lrn = mlr::makeLearner("classif.rpart", predict.type = "prob")
  tsk = mlr::makeClassifTask(data = iris, target = "Species")
  mod = mlr::train(lrn, tsk)
  pred = Predictor$new(mod, data = iris, y = iris$Species == "setosa", class = "setosa")
  expect_warning({var.imp = FeatureImp$new(pred, loss = "mae")}, "Model error is 0")
  expect_equal(var.imp$compare, "difference")
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_false("data.table" %in% class(dat))
  expect_equal(colnames(dat), expected_colnames)
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
})

test_that("FeatureImp works for single output and function as loss", {
  
  var.imp = FeatureImp$new(predictor1, loss = Metrics::mse)
  dat = var.imp$results
  expect_class(dat, "data.frame")
  # Making sure the result is sorted by decreasing importance
  expect_equal(dat$importance, dat[order(dat$importance, decreasing = TRUE),]$importance)
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))  
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
  
})

test_that("FeatureImp works for multiple output",{
  var.imp = FeatureImp$new(predictor2, loss = "ce")
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))  
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
})


test_that("FeatureImp fails without target vector",{
  predictor2 = Predictor$new(f, data = X, predict.fun = predict.fun)
  expect_error(FeatureImp$new(predictor2, loss = "ce"))
})

test_that("Works for different repetitions.",{
  var.imp = FeatureImp$new(predictor1, loss = "mse", n.repetitions = 2)
  dat = var.imp$results
  expect_class(dat, "data.frame")
})


test_that("Model receives data.frame without additional columns", {
  # https://stackoverflow.com/questions/51980808/r-plotting-importance-feature-using-featureimpnew
  library(mlr)
  library(ranger)
  data("iris")
  tsk = mlr::makeClassifTask(data = iris, target = "Species")
  lrn = mlr::makeLearner("classif.ranger",predict.type = "prob")
  mod = mlr:::train(lrn, tsk)
  X = iris[which(names(iris) != "Species")]
  predictor = Predictor$new(mod, data = X, y = iris$Species)
  imp = FeatureImp$new(predictor, loss = "ce")
  expect_r6(imp)
})

set.seed(12)
X = data.frame(x1 = 1:100)
X$x2 = X$x1 + rnorm(100)
X$x3 = rnorm(100)
X$x4 = sample(c(0,1), size = 100, replace = TRUE)
X$x5 = factor(sample(c(1,2,3), size = 100, replace = TRUE))
y = X[,1] + X[,2] + rnorm(10, 0, 0.1)
pred.fun = function(newdata){
  newdata[,1] + newdata[,2]
}
pred = Predictor$new(data = X, predict.fun = pred.fun, y = y)

test_that("Feature Importance 0", {
 fimp = FeatureImp$new(pred, loss = "mae", n.repetitions = 3)
 expect_equal(fimp$results$importance[3], 1) 
})


test_that("Feature Importance 0", {
 fimp = FeatureImp$new(pred, loss = "mae", n.repetitions = 3)
 expect_equal(fimp$results$importance[3], 1)
})

test_that("Conditional Feature Importance", {
  predictor1 = Predictor$new(data = X, y = y, predict.fun = pred.fun, conditional = TRUE)
  var.imp = FeatureImp$new(predictor1, loss = "mse", conditional = TRUE)
  dat = var.imp$results
  expect_class(dat, "data.frame")
  expect_false("data.table" %in% class(dat))
  expect_equal(colnames(dat), expected_colnames)
  expect_equal(nrow(dat), ncol(X))
  p = plot(var.imp)
  expect_s3_class(p, c("gg", "ggplot"))
  p
  p = plot(var.imp, sort = FALSE)
  expect_s3_class(p, c("gg", "ggplot"))
  p
  
  p = var.imp$plot()
  expect_s3_class(p, c("gg", "ggplot"))
  p
})


