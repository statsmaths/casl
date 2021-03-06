% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ch10-implementation.R
\name{casl_sparse_irwls}
\alias{casl_sparse_irwls}
\title{Compute GLM regression with a sparse data matrix.}
\usage{
casl_sparse_irwls(X, y, family = stats::binomial(), maxit = 25L,
  tol = 1e-08)
}
\arguments{
\item{X}{A sparse or dense numeric matrix.}

\item{y}{The numeric response vector.}

\item{family}{Instance of an R `family` object.}

\item{maxit}{Integer maximum number of iterations.}

\item{tol}{Numeric tolerance parameter.}
}
\value{
The estimated regression vector.
}
\description{
Compute GLM regression with a sparse data matrix.
}
\references{
Taylor Arnold, Michael Kane, and Bryan Lewis.
\emph{A Computational Approach to Statistical Learning}.
Chapman & Hall/CRC Texts in Statistical Science, 2019.
}
\author{
Taylor Arnold, Michael Kane, Bryan Lewis.
}
