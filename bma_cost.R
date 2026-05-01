# bma_cost.R — adapted from TUNE's bma-package/bma_cost.R for our local data.
#
# TUNE original: regressed `x64 ~ .` over a sliced column range. That assumes
# x64 is the binary target — true for TUNE's old data file but NOT for our
# data/training_rescueRobot_25600_64.csv, which has the binary target in a
# `hazard` column (0/1) and continuous features x1..x64.
#
# This adaptation: regress `hazard ~ x1 + x2 + ... + x{vars}` so the response
# is the actual binary outcome, matching what run_rq3.py does for BMA and
# Stacking. Sampler choice (MCMC vs BAS) and other settings (5000 iters,
# bic.prior, uniform model prior, 200s timeout) are kept verbatim from TUNE.

suppressMessages(library(BMA))
suppressMessages(library(MASS))
suppressMessages(library(BAS))
suppressMessages(library(argparse))
suppressMessages(library(R.utils))

defaultW <- getOption('warn')
options(warn = -1)

parser <- ArgumentParser()
parser$add_argument('-s', '--sample', type='integer', default=200,
                    help='Number of observations [default 200]',
                    metavar='sample')
parser$add_argument('-v', '--vars', type='integer', default=2,
                    help='Number of explanatory variables [default 2]',
                    metavar='vars')
parser$add_argument('-m', '--method', type='character', default='MCMC',
                    help='Sampling method (MCMC | BAS | MCMC+BAS) [default MCMC]',
                    metavar='method')
args <- parser$parse_args()
SAMPLE <- args$sample
VARS   <- args$vars
METHOD <- args$method

df <- read.csv('data/training_rescueRobot_25600_64.csv')

# Build a model frame: hazard (0/1 binary target) plus the first VARS feature
# columns (x1, x2, ..., x{VARS}). bas.glm needs the response in the data frame.
feature_cols <- paste0('x', seq_len(VARS))
model_df <- df[seq_len(SAMPLE), c(feature_cols, 'hazard')]

bma_fit <- function(model_df, selected_method = 'MCMC') {
  bas.glm(hazard ~ ., data = model_df,
          method = selected_method, MCMC.iterations = 5000,
          betaprior = bic.prior(),
          family = binomial(link = 'logit'),
          modelprior = uniform())
}

run_bma <- function(model_df, sample_size, vars, selected_method = 'MCMC') {
  t0 <- Sys.time()
  reg <- withTimeout(
    bma_fit(model_df, selected_method),
    timeout = 200.0, elapsed = 200.0, onTimeout = 'silent')
  time_diff <- as.numeric(Sys.time() - t0, units = 'secs')
  cat(selected_method, vars, sample_size, time_diff, sep = ' ')
}

run_bma(model_df, SAMPLE, VARS, METHOD)

options(warn = defaultW)
