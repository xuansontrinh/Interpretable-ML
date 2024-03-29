#### Functions as input for slickEcr ####

#' Compute fitness values
#'
#' Returns a matrix of three rows, one for the distance of the prediction to
#' the desired target, one for the distance to x.interest and
#' one for the number of changed features.
#'
#' @section Arguments:
#' \describe{
#' \item{x: }{(data.frame)\cr Data frame of candidates.}
#' \item{x.interest: }{(data.frame)\cr Data frame with one row as instance to
#' be explained. }
#' \item{target: }{numeric(1)|numeric(2) \cr Desired outcome either a single numeric or
#' a vector of two numerics in order to define an interval as desired outcome.}
#' \item{predictor: }{(Predictor)\cr
#' The object that holds the machine learning model and the data.}
#' \item{range: }{(numeric)\cr Vector of ranges for numeric features.
#' Must have same ordering as columns in x and x.interest.}
#' }
#' @return (matrix)
fitness_fun = function(x, x.interest, target, predictor, train.data, range = NULL,
  track.infeas = TRUE, identical.strategy = FALSE, k = 1, weights = NULL) {
  assertIntegerish(k, null.ok = FALSE)
  assertDataFrame(x)
  assertDataFrame(x.interest, nrows = 1, any.missing = FALSE)
  assertNumeric(target, min.len = 1, max.len = 2)
  assertNumeric(range, lower = 0, finite = TRUE, any.missing = TRUE,
    len = ncol(x.interest), null.ok = TRUE)
  assertNumeric(weights, null.ok = TRUE, finite = TRUE, any.missing = FALSE,
    len = k)
  if (!(is.null(range)) && !(all(names(x.interest) == names(range)))) {
    stop("range for gower distance needs same order and feature names as
      'x.interest'")
  }

  if (any(grepl("use.orig", names(x)))) {
    x = x[, -grep("use.orig", x = names(x))]
  }
  assert_data_frame(x, ncols = ncol(x.interest))

  if(!all(names(x) == names(x.interest))) {
    stop("x.interest and x need same column ordering and names")
  }
  equal.type = all(mapply(x, x.interest,
    FUN = function(x1, x2) {class(x1) == class(x2)}))
  if (!equal.type) {
    stop("x.interest and x need same feature types")
  }

  # Objective Functions
  pred = predictor$predict(newdata = x)[[1]]
  q1 = vapply(pred, numeric(1), FUN =  function(x) min(abs(x - target)))
  q1 = ifelse(length(target) == 2 & (pred >= target[1]) & (pred <= target[2]),
    0, q1)
  q2 = StatMatch::gower.dist(data.x = x.interest, data.y = x, rngs = range,
    KR.corr = FALSE)[1,]

  if (identical.strategy) {
    feature.types = predictor$data$feature.types
    x.interest.rep = x.interest[rep(row.names(x.interest), nrow(x)),]
    notid = x != x.interest.rep
    id.num = predictor$data$feature.types == "numerical"
    if (sum(id.num) > 0) {
      notid[, id.num] =  abs(x[, id.num] - x.interest.rep[, id.num]) >= sqrt(.Machine$double.eps)
    }
    q3 = rowSums(notid)
  } else {
    q3 = rowSums(x != x.interest[rep(row.names(x.interest), nrow(x)),])
  }
  if (track.infeas) {
    q4 = apply(StatMatch::gower.dist(data.x = train.data, data.y = x, rngs = range,
      KR.corr = FALSE), MARGIN = 2, FUN = function(dist) {
        d = sort(dist)[1:k]
        if (length(d) == k) {
          if (!is.null(weights)) {
            d = weighted.mean(d, w = weights)
          } else {
            d = mean(d)
          }
        }
       return(d)
      })
    fitness = mapply(function(a, b, c, d) {
      c(a, b, c, d)
    }, q1, q2, q3, q4)
  } else {
    fitness = mapply(function(a, b, c) {
      c(a, b, c)
    }, q1, q2, q3)
  }
  return(fitness)
}


#' Select diverse candidates for new generation
#'
#' Replaces ecr::replaceMuPlusLambda,
#' and ecr::selNondom, because the functions are not able
#' to compute crowding distance also for feature space.
#'
#'@section Arguments:
#' \describe{
#' \item{control: }{(ecr_control)\cr Object that holds information on
#' algorithm parameters.}
#' \item{population: }{(list)\cr Object that holds population.}
#' \item{offspring: }{(list)\cr Object that holds offspring.}
#' \item{fitness: }{(matrix)\cr Matrix of fitness values. Each column corresponds to
#' element of `population`.}
#' \item{fitness.offspring: }{(matrix)\cr Matrix of fitness values. Each column corresponds to
#' element of `offspring`.}
#' \item{n.select: }{(numeric(1))\cr Number of candidates, which are selected
#' for next generation.}
#' \item{epsilon: }{(numeric(1))\cr Soft constraint. If chosen, candidates, whose
#'   distance between their prediction and target exceeds epsilon, are penalized.}
#' \item{extract.duplicate}{(logical(1)) Whether to penalize duplicates and
#' move them to the last front in nondominated sorting. Default TRUE.}
#' \item{candidates}{(data.frame)\cr Data frame from which `n.select` best and most
#' diverse ones are chosen.}
#' }
#' @return (list)
select_diverse = function (control, population, offspring, fitness,
  fitness.offspring) {
  assertClass(control, "ecr_control")
  assertClass(control$selectForSurvival, "ecr_selector")
  assertList(population)
  assertList(offspring)
  assertMatrix(fitness, ncols = length(population), any.missing = FALSE,
    min.rows = 1L)
  assertMatrix(fitness.offspring, ncols = length(offspring), any.missing = FALSE,
    min.rows = 1L)
  merged.pop = c(population, offspring)
  merged.pop.df = mosmafs::listToDf(merged.pop, control$task$par.set)
  merged.pop.df[, grepl("use.orig", names(merged.pop.df))] = NULL
  merged.fit = cbind(fitness, fitness.offspring)
  surv.idx = control$selectForSurvival(merged.fit, n.select = length(population),
    merged.pop.df)
  fitness = merged.fit[, surv.idx, drop = FALSE]
  browser(expr = anyNA(fitness))
  fitness = BBmisc::addClasses(fitness, "ecr_fitness_matrix")
  fitness = BBmisc::setAttribute(fitness, "minimize", control$task$minimize)
  return(list(population = merged.pop[surv.idx], fitness = fitness))
}

#' @rdname select_diverse
select_nondom = ecr::makeSelector(
  selector = function(fitness, n.select, candidates,
    epsilon = Inf,
    extract.duplicates = TRUE) {

    assert_number(n.select)
    if (n.select > ncol(fitness)-1) {
      stop("'n.select' must be smaller than 'ncol(fitness)'")
    }
    assert_matrix(fitness, ncols = nrow(candidates))
    assert_numeric(epsilon, null.ok = TRUE)
    assert_logical(extract.duplicates)

    infeasible.idx = which(fitness[1,] > epsilon)
    order.infeasible = order(fitness[1, infeasible.idx])


    # if duplicates should be substracted, get indices of unique elements
    if (extract.duplicates) {
      duplicated.idx = which(duplicated(t(fitness)))
    } else {
      duplicated.idx = integer(0)
    }

    # normalize objectives
    mean.f = rowMeans(fitness)
    sd.f = apply(fitness, 1, sd)
    sd.f[sd.f == 0] = 1
    # if any sd 0, transform to 1
    fitness = apply(fitness, 2, function(x) {
      (x-mean.f)/sd.f
    })

    # get domination layers
    nondom.layers = ecr::doNondominatedSorting(fitness)
    ranks = nondom.layers$ranks

    # get number of domination layers
    max.rank = max(ranks)

    # change domination layer of infeasible solutions
    i = 0
    if (!is.null(infeasible.idx)) {
      for (inf.id in infeasible.idx[order.infeasible]) {
        ranks[inf.id] = max.rank + i
        i = i + 1
      }
    }
    max.rank = max(ranks)

    # Extract duplicated points, if number of unique elements does
    # exceed n.select
    if (extract.duplicates) {
      ranks[duplicated.idx] = max.rank + 1
    }

    # storage for indizes of selected individuals
    new.pop.idxs = integer()

    # Count number of points per domination layer and cumulate their numbers
    front.len = table(ranks)
    front.len.cumsum = cumsum(front.len)

    # Get first front, that does exceed n.select
    front.first.nonfit = as.numeric(names(front.len.cumsum)[
      BBmisc::which.first(front.len.cumsum > n.select)[[1]]])

    if (front.first.nonfit > 1) {
      new.pop.idxs = (1:length(ranks))[ranks < front.first.nonfit]
    }
    # how many points to select by second criterion, i.e., crowding distance?
    n.diff = n.select - length(new.pop.idxs)

    if (n.diff > 0L) {
      idxs.first.nonfit = which(ranks == front.first.nonfit)
      #if (vers == 1) {
        cds = computeCrowdingDistanceR(as.matrix(fitness[, idxs.first.nonfit]),
          candidates[idxs.first.nonfit,])
      #}
      # if (vers == 3) {
      #   cds = ecr::computeCrowdingDistance(as.matrix(fitness[, idxs.first.nonfit]))
      # }
      idxs2 = order(cds, decreasing = TRUE)[1:n.diff]
      new.pop.idxs = c(new.pop.idxs, idxs.first.nonfit[idxs2])
    }

    return(new.pop.idxs)
  },
  supported.objectives = "multi-objective")

#' @rdname select_diverse
computeCrowdingDistanceR = function(fitness, candidates) {
  assertMatrix(fitness, mode = "numeric", any.missing = FALSE, all.missing = FALSE)
  assertDataFrame(candidates, nrows = ncol(fitness))

  n = ncol(fitness)
  max = apply(fitness, 1, max)
  min = apply(fitness, 1, min)
  dim = nrow(fitness)
  ods = numeric(n)
  dds = numeric(n)
  cds = numeric(n)

  g.dist = StatMatch::gower.dist(candidates, KR.corr = FALSE)

  for (i in c(1,2,3,4)) {

    # get the order of the points when sorted according to the i-th objective
    ord = order(fitness[i, ])

    # set the extreme values to Inf
    ods[ord[1]] = Inf
    ods[ord[n]] = Inf
    dds[ord[1]] = Inf
    dds[ord[n]] = Inf

    # update the remaining crowding numbers
    if (n > 2L) {
      for (j in 2:(n - 1L)) {

        ods[ord[j]] = ods[ord[j]] +
          ((fitness[i, ord[j + 1L]] - fitness[i, ord[j - 1L]])/(max[i]-min[i]))

        dds[ord[j]] = dds[ord[j]] +
          g.dist[ord[j], ord[j-1]] +
          g.dist[ord[j], ord[j+1]]

      }
    }
  }

  cds = rank(ods, ties.method = "random") +
    rank(dds, ties.method = "random")
  cds = jitter(cds, factor = 1)
  return(cds)
}



#' Mutator that uses conditional distributions.
#'
#' @section Arguments:
#' \describe{
#' \item{ind: }{(numeric(1)/character(1))\cr Named vector of feature position.}
#' \item{X: }{(data table)\cr Candidate of which features are mutated.}
#' \item{pred: }{(Predictor)\cr The object that holds the machine learning model and the data.}
#' \item{param.set: }{(ParamSet)\cr Parameter set extracted from observed data.}
#' }
#' @return (numeric(1)/character(1))
mutConDens = ecr::makeMutator(function(ind, X, pred, param.set,...) {
  a = names(ind)
  val = pred$conditionals[[a]]$csample(X = X, size = 1, type = "parametric")[[1]]
  # num.feat = names(pred$data$feature.types)[pred$data$feature.types == "numerical"]
  # for (col in num.feat) X[,col] = as.numeric(X[,col])
  # X = X[, duration := as.numeric(duration)]
  # sapply(dt, class)
  # pred$conditionals[[a]]$csample_parametric(X = X, size = 1)
  # Warning if ordinal feature is defined as numeric, because> ties occur in datasets
  # https://stackoverflow.com/questions/56861001/how-to-suppress-warnings-from-statsregularize-values
  if (param.set$pars[[a]]$type == "discrete") {
    val = as.character(val)
  }
  if (param.set$pars[[a]]$type == "numeric") {
    lower = param.set$pars[[a]]$lower
    upper = param.set$pars[[a]]$upper
    val = pmin(pmax(lower, val), upper)
  }
  if (param.set$pars[[a]]$type == "integer") {
    lower = param.set$pars[[a]]$lower
    upper = param.set$pars[[a]]$upper
    val = as.integer(pmin(pmax(lower, val), upper))
  }
  ind[a] = val
  return(ind)
}, supported = "custom")

mutInd = ecr:::makeMutator(function(ind, ...) {
  ind
}, supported = "custom")
