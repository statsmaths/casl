% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ch06-additive-models.R
\name{casl_am_ls_basis}
\alias{casl_am_ls_basis}
\title{Compute additive model design matrix.}
\usage{
casl_am_ls_basis(X, order, nknots)
}
\arguments{
\item{X}{A numeric data matrix.}

\item{order}{Order of the polynomial.}

\item{nknots}{Number of knots in the power basis.}
}
\value{
A list of smoothing spline objects; one for each column
in X.
}
\description{
Compute additive model design matrix.
}
\references{
Taylor Arnold, Michael Kane, and Bryan Lewis.
\emph{A Computational Approach to Statistical Learning}.
Chapman & Hall/CRC Texts in Statistical Science, 2019.
}
\author{
Taylor Arnold, Michael Kane, Bryan Lewis.
}
