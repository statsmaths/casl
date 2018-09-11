#' Compute square kernel matrix for polynomial kernel.
#'
#' @param X A numeric data matrix.
#' @param d Integer degree of the polynomial.
#' @param c Numeric constant to modify the kernel.
#'
#' @return A square kernel matrix with the nrow(X) rows and columns.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_util_kernel_poly <-
function(X, d=2L, c=0)
{
  cross <- tcrossprod(X, X)
  M <- (cross + c)^d
  M
}

#' Compute square kernel matrix for a radial kernel.
#'
#' @param X A numeric data matrix.
#' @param gamma Positive tuning parameter to set kernel shape.
#'
#' @return A square kernel matrix with the nrow(X) rows and columns.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_util_kernel_radial <-
function(X, gamma=1)
{
  d <- as.matrix(stats::dist(X))^2
  M <- exp(-1 * gamma * d)
  M
}

#' Calculate normalized kernel matrix.
#'
#' @param M A raw kernel matrix.
#'
#' @return A normalized version of the kernel matrix.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_util_kernel_norm <-
function(M)
{
  n <- ncol(M)
  ones <- matrix(1 / n, ncol = n, nrow = n)
  Ms <- M - 2 * ones %*% M  + ones %*% M %*% ones
  Ms
}

#' Compute kernel version of PCA matrix.
#'
#' @param X A numeric data matrix.
#' @param k Integer number of components to return.
#' @param kfun The kernel function.
#' @param ... Other options passed to the kernel function.
#'
#' @return The nrow(X)-by-k kernel PCA matrix.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_kernel_pca <-
function(X, k=2L, kfun=casl_util_kernel_poly, ...)
{
  M <- casl_util_kernel_norm(kfun(X, ...))
  e <- irlba::partial_eigen(M, n = k)
  pc <- e$vectors %*% diag(sqrt(e$values))
  pc[]
}

#' Compute symmetric k-nearest neighbors similarity matrix.
#'
#' @param X A numeric data matrix.
#' @param k Number of nearest neighbors to consider.
#'
#' @return A square similarity matrix with nrow(X) rows & columns.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_util_knn_sim <-
function(X, k=10L)
{
  d <- as.matrix(stats::dist(X, upper = TRUE))^2
  N <- apply(d, 1L, function(v) v <= sort(v)[k + 1L])
  S <- (N + t(N) >= 2)
  diag(S) <- 0
  S
}

#' Perform spectral clustering using a similarity matrix.
#'
#' @param S A symmetric similarity matrix.
#' @param k Number of components to return.
#'
#' @return A matrix with nrow(S) rows and k columns.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_spectral_clust <-
function(S, k=1)
{
  D <- diag(apply(S, 1, sum))
  L <- D - S
  e <- eigen(L)
  Z <- e$vector[,rev(which(e$value > 1e-8))[seq_len(k)]]
  Z
}

#' Compute t-SNE probability scores.
#'
#' @param X A numeric data matrix.
#' @param svals Desired variances of the Gaussian densities.
#' @param norm Logical. Should probabilities be normalized to 1.
#'
#' @return A matrix of probability densities.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_tsne_p <-
function(X, svals, norm=TRUE)
{

  d <- exp(-1 * as.matrix(stats::dist(X))^2 / (2 * svals))
  diag(d) <- 0
  P <- t(t(d) / apply(d, 2, sum))
  if (norm) P <- (P + t(P)) / (2 * nrow(X))

  P
}

#' Compute t-SNE variance values.
#'
#' @param X A numeric data matrix.
#' @param perplexity Desired perplexity score for all variables.
#'
#' @return A vector of estimated variances for each variable.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_tsne_sigma <-
function(X, perplexity)
{
  fn <- function(s)
  {
    P <- casl_tsne_p(X, s, norm = FALSE)
    perp <- 2^(-1 * apply(P, 2, sum))
    sum((perp - perplexity)^2)
  }

  svals <- stats::optim(stats::runif(nrow(X)), fn)$par
  svals
}

#' Compute t-SNE variance values.
#'
#' @param X A numeric data matrix.
#' @param perplexity Desired perplexity score for all variables.
#' @param k Dimensionality of the output.
#' @param iter Number of iterations to perform.
#' @param rho A positive numeric learning rate.
#'
#' @return An nrow(X) by k matrix of t-SNE embeddings.
#'
#' @author Taylor Arnold, Michael Kane, Bryan Lewis.
#'
#' @references
#'
#' Taylor Arnold, Michael Kane, and Bryan Lewis.
#' \emph{A Computational Approach to Statistical Learning}.
#' Chapman & Hall/CRC Texts in Statistical Science, 2019.
#'
#' @export
casl_tsne <-
function(X, perplexity, k=2L, iter=1000, rho=0.1)
{
  svals <- casl_tsne_sigma(X, perplexity)
  Y <- matrix(stats::runif(nrow(X) * k), ncol = 2L)
  P <- casl_tsne_p(X, perplexity)

  for (k in seq_len(iter))
  {
    d <- as.matrix(stats::dist(Y))^2
    d <- (1 + d)^(-1)
    diag(d) <- 0
    Q <- d / sum(d)
    fct <- (P - Q) * d
    gd <- Y * 0
    for (i in seq_len(nrow(Y)))
    {
      gd[i,] <- apply(t(t(Y) - Y[i, ]) * fct[i, ], 2, sum)
    }

    Y <- Y - gd * rho
  }

  Y
}
