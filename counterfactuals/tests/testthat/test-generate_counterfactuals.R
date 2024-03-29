context("generate_counterfactuals")

test_that("fitness function", {
  library("mlrCPO")
  x.interest = data.frame(one = "c", two = 25, three = 3, stringsAsFactors = TRUE)
  x = data.frame(one = "d", two = 20, three = 3, stringsAsFactors = TRUE)

  test_predictor = R6Class(classname = "test_predictor",
    public = list(
      predict = function(newdata) {
        return(1 + x$two * 1 + x$three * 1)
      }))
  test_pred = test_predictor$new()

  classes = c("a", "b", "c", "d")
  ps = pSS(
    one:discrete [classes],
    two:integer [10, 30],
    three:numeric [0, 5]
  )
  target = 30

  expect_error(fitness_fun(x, x.interest, target, test_pred, train.data = x),
    "x and y have different levels")
  x.interest$one = factor(x.interest$one, levels = classes)
  x$one = as.character(x$one)
  expect_error(fitness_fun(x, x.interest, target, test_pred),
    "x.interest and x need same feature types")
  x$one = factor(x$one, levels = classes)

  expect_error(fitness_fun(x[,1:2], x.interest, target, test_pred),
    "'x' failed: Must have exactly 3 cols, but has 2 cols")

  x.re = x # renamed x column
  names(x.re)[1] = "eins"
  expect_error(fitness_fun(x.re, x.interest, target, test_pred),
    "x.interest and x need same column ordering and names")

  fit.val = fitness_fun(x, x.interest, target, test_pred, track.infeas = FALSE, train.data = x)

  expect_matrix(fit.val, nrows = 3, ncols = 1, any.missing = FALSE)

  expect_identical(fit.val[1, 1], abs(test_pred$predict(x) - target))

  expect_identical(fit.val[3, 1], 2)

  expect_identical(fit.val[2, 1], (1/1 + 5/5 + 0)/3)

  fit.val2 = fitness_fun(x, x.interest, target = c(30, 35), test_pred,
    track.infeas = FALSE, train.data = x)

  expect_identical(fit.val2, fit.val)

  fit.val3 = fitness_fun(x, x.interest, target = c(25, 35), test_pred,
    range = c("one" = NA, "two" = 10, "three" = 1), track.infeas = FALSE, train.data = x)
  expect_identical(fit.val3[3, ], fit.val3[3,])
  expect_identical(fit.val3[2, ], (1/1 + 5/10 + 0)/3)
  expect_identical(fit.val3[1, ], 1)

  fit.val4 = fitness_fun(x, x.interest, target = c(24, 35), test_pred,
    range = c("one" = NA, "two" = 10, "three" = 1), track.infeas = FALSE, train.data = x)
  expect_identical(fit.val4[1, ], 0)
  expect_identical(fit.val3[2:3, ], fit.val4[2:3, ])

  fit.val5 = fitness_fun(x, x.interest, target = 23, test_pred, track.infeas = FALSE, train.data = x)
  expect_identical(fit.val5[1, ], 1)
  expect_identical(fit.val2[2:3, ], fit.val5[2:3, ])

  # use.orig is removed?
  x$use.orig1 = TRUE
  x$use.orig2 = FALSE

  expect_identical(fitness_fun(x, x.interest, target = 23, test_pred,
    track.infeas = FALSE, train.data = x),
    fit.val5)

  # infinity
  expect_identical(fitness_fun(x, x.interest, target = c(-Inf, 23), test_pred,
    track.infeas = FALSE, train.data = x),
    fit.val5)
  fit.val6 = fitness_fun(x, x.interest, target = c(23, Inf), test_pred,
    track.infeas = FALSE, train.data = x)
  expect_identical(fit.val6[2:3, ], fit.val5[2:3,])
  expect_identical(fit.val6[1, ], 0)

  # range
  expect_error(fitness_fun(x, x.interest, target = c(24, 35), test_pred,
    range = c("two" = 10, "one" = NA, "three" = 1), track.infeas = FALSE, train.data = x),
    "range for gower distance needs same order and feature names as
      'x.interest'")

  # identical strategy
  fit.val.ind = fitness_fun(x, x.interest, target, test_pred, track.infeas = FALSE, train.data = x,
    identical.strategy = TRUE)
  expect_true(all(fit.val == fit.val.ind))
})



test_that("select_nondom", {
  fitness = matrix(c(0.5, 0.9, 0.7, 1, 0.5, 0.8, 0, 0,
    0.7, 0.3, 0.5, 0.7, 0.7, 1, 0, 0,
    4, 3, 2, 7, 4, 5, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0), nrow = 4, ncol = 8, byrow = TRUE)
  # nondom, nondom, nondom, dom, dup1, dom, nondom, dup7
  population = data.frame("a" = 1, "b" = 2, "c" = "a")
  population = population[rep(seq_len(nrow(population)), each=8),]

  expect_true(all(c(7, 8) %in% select_nondom(fitness, n.select = 4, population, extract.duplicates = FALSE)))
  expect_true(7 %in% select_nondom(fitness, n.select = 4, population))
  expect_true(all(select_nondom(fitness, n.select = 4, population) %in% c(7, 1, 2, 3)))
  expect_true(all(select_nondom(fitness, n.select = 2, population, extract.duplicates = FALSE) %in%
      c(7, 8)))
  expect_true(all(select_nondom(fitness, n.select = 4, population,
    extract.duplicates = FALSE) %in% c(7, 8, 1, 2, 3, 5)))

  expect_equal(select_nondom(fitness, n.select = 4, population, epsilon = 0.3),
    c(1, 3, 6, 7))

  fitness = matrix(c(1, 2, 3, 0, 0, 0), nrow = 3, ncol = 2, byrow = TRUE)
  population = data.frame("a" = 1, "b" = 2, "c" = "a")
  population = population[rep(seq_len(nrow(population)), each=2),]
  expect_error(select_nondom(fitness, n.select = 2, population),
    "'n.select' must be smaller ")

})

# test_that("mutConDens", {
#   discval = c("a", "b", "c")
#   test.df = data.frame(num = runif(10, 0, 1),
#     char = sample(discval, size = 10, replace = TRUE),
#     fact = factor(sample(discval, size = 10, replace = TRUE), levels = c(discval)),
#     int = sample(1:20, size = 10), stringsAsFactors = FALSE)
#   ps = ParamHelpers::makeParamSet(params = make_paramlist(test.df))
#
#   mutConDens(ind = mosmafs::listToDf(sampleValues(ps, n = 1L, discrete.names = TRUE), par.set = ps),
#     )
# })

