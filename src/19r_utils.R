# 19r_utils.R — helpers for `19r_wis_bam.ipynb` (2026-11-03, user request).
#
# 19r regresses log-scale WIS per (series, origin, age bin) on the scenario and contact-model features, one fit per
# horizon, with mgcv `bam` under three families (lognormal = gaussian on log WIS, gamma, Tweedie). Here: the
# covariate coding of the 22 series, the formula, the fit/AIC/draw helpers, the Tweedie brute-force density and
# simulator behind the notebook's parameterisation check and posterior predictive check, and 9j's periods.
#
# Tweedie with 1 < p < 2 is the compound Poisson–gamma: N ~ Poisson(lambda), Y = sum of N Gamma(a, scale s), with
#     lambda = mu^(2-p) / (phi (2-p)),   a = (2-p)/(p-1),   s = phi (p-1) mu^(p-1),
# so E[Y] = mu and Var[Y] = phi mu^p, which is mgcv's `tw()` / `ldTweedie(y, mu, p, phi)` parameterisation.

suppressPackageStartupMessages({
  library(mgcv)
  library(data.table)
})
# `bam(nthreads > 1)` calls BLAS from inside its OpenMP threads, and this image's OpenBLAS (pthreads build) then
# prints "Detect OpenMP Loop and this application may hang" on every call. One BLAS thread removes the hazard; mgcv's
# own threads are unaffected (measured: 2.3 s at nthreads = 17 vs 2.4 s at 1 for one gamma fit, so neither matters).
RhpcBLASctl::blas_set_num_threads(1)

# ---- Tweedie: reference density and simulator ---------------------------------------------------------------
# The density of y > 0 as the plain series over n >= 1 of Pois(n; lambda) * Gamma(y; n a, scale s). Slow but
# obviously right: the notebook checks mgcv's `ldTweedie` against it.
dtweedie_log_brute <- function(y, mu, disp, pw, nmax = 20000) {
  lambda <- mu^(2 - pw) / (disp * (2 - pw)); a <- (2 - pw) / (pw - 1); s <- disp * (pw - 1) * mu^(pw - 1)
  n <- 1:nmax
  lt <- dpois(n, lambda, log = TRUE) + dgamma(y, shape = n * a, scale = s, log = TRUE)
  m <- max(lt); m + log(sum(exp(lt - m)))
}

rtweedie <- function(k, mu, disp, pw) {
  lambda <- mu^(2 - pw) / (disp * (2 - pw)); a <- (2 - pw) / (pw - 1); s <- disp * (pw - 1) * mu^(pw - 1)
  N <- rpois(k, lambda)
  rgamma(k, shape = N * a, scale = s)   # shape 0 gives exactly 0, the point mass
}

# ---- design -------------------------------------------------------------------------------------------------
# One row per series. X_data: future / frozen contacts, the variant renewal and their interactions. X_model (0 for
# the two no-contact series): duration weighting, neighbourhood NGM, their interaction, diagonal NGM. The
# reference is no contact – no variant, and `unweighted-negbin|mean` among the contact models. Interactions are
# PRODUCT COLUMNS, not `a:b` terms, because `s(origin, by = x, bs = "re")` needs a numeric `by`; using the same
# columns in the parametric part gives each fixed term and its random slope one name.
SCENARIOS <- c("no contact", "contact", "contact (frozen)", "variant", "variant + contact",
               "variant + contact (frozen)")
X_TERMS <- c("ctc_future", "ctc_frozen", "variant", "ctc_future_variant", "ctc_frozen_variant",
             "duration", "nbhd", "duration_nbhd", "diagonal")

add_design <- function(d) {
  stopifnot(all(d$scenario %in% SCENARIOS))
  base <- sub("-variant$", "", d$label)           # `label` carries no `@frozen`; strip the variant suffix
  degree <- sub("\\|.*$", "", base); ngm <- sub("^.*\\|", "", base)
  stopifnot(all(degree %in% c("no-contact", "unweighted-negbin", "weighted-hweibull")),
            all(ngm %in% c("null", "mean", "neighbourhood", "mean-diagonal")))
  d[, `:=`(
    ctc_future = as.integer(scenario %in% c("contact", "variant + contact")),
    ctc_frozen = as.integer(scenario %in% c("contact (frozen)", "variant + contact (frozen)")),
    variant    = as.integer(scenario %in% c("variant", "variant + contact", "variant + contact (frozen)")),
    duration   = as.integer(degree == "weighted-hweibull"),
    nbhd       = as.integer(ngm == "neighbourhood"),
    diagonal   = as.integer(ngm == "mean-diagonal")
  )]
  d[, `:=`(ctc_future_variant = ctc_future * variant, ctc_frozen_variant = ctc_frozen * variant,
           duration_nbhd = duration * nbhd)]
  # the contact-model indicators must be 0 exactly where there is no contact model
  stopifnot(all((d$ctc_future + d$ctc_frozen == 0) == (degree == "no-contact")))
  d
}

# Fixed effects for all 9 terms, random intercepts for origin, age group and origin × age, and a random slope by
# origin on every one of the 9 terms. bam's `re` slopes are independent of each other and of the intercept.
wis_formula <- function(fam, slopes = TRUE) {
  lhs <- if (fam == "lognormal") "log(wis)" else "wis"
  re <- c('s(origin, bs = "re")', 's(age_group, bs = "re")', 's(origin_age, bs = "re")')
  if (slopes) re <- c(re, sprintf('s(origin, by = %s, bs = "re")', X_TERMS))
  as.formula(paste(lhs, "~", paste(c(X_TERMS, re), collapse = " + ")))
}

FAMILIES <- c("lognormal", "gamma", "tweedie")
wis_family <- function(fam) switch(fam, lognormal = gaussian(), gamma = Gamma(link = "log"), tweedie = tw(link = "log"),
                                   stop("unknown family ", fam))

fit_wis_bam <- function(d, fam, slopes = TRUE, nthreads = max(1, parallel::detectCores() - 1)) {
  bam(wis_formula(fam, slopes), data = d, family = wis_family(fam), method = "fREML", discrete = TRUE,
      nthreads = nthreads)
}

tweedie_p <- function(fit) if (inherits(fit$family, "extended.family") && grepl("^Tweedie", fit$family$family))
  fit$family$getTheta(TRUE) else NA_real_

# AIC for the density of `wis` itself, so the three families compare on one scale: the lognormal fit models
# log(wis), whose density differs from that of wis by the Jacobian 1/y, i.e. log-likelihood − Σ log y.
aic_on_wis <- function(fit, fam, y) AIC(fit) + if (fam == "lognormal") 2 * sum(log(y)) else 0

# ---- draws --------------------------------------------------------------------------------------------------
# Draws from the Bayesian posterior approximation N(coef, Vp) (conditional on the smoothing parameters).
coef_draws <- function(fit, n = 4000, seed = 1236) {
  set.seed(seed)
  dr <- rmvn(n, coef(fit), vcov(fit))
  colnames(dr) <- names(coef(fit))
  dr
}

# Multiplicative effects on WIS: the 9 terms, plus the contrasts the coefficients only imply.
fixed_effect_draws <- function(dr) {
  stopifnot(all(X_TERMS %in% colnames(dr)))
  b <- function(t) dr[, t]
  out <- data.table(
    ctc_future = b("ctc_future"), ctc_frozen = b("ctc_frozen"), variant = b("variant"),
    ctc_future_variant = b("ctc_future_variant"), ctc_frozen_variant = b("ctc_frozen_variant"),
    duration = b("duration"), nbhd = b("nbhd"), duration_nbhd = b("duration_nbhd"), diagonal = b("diagonal"),
    # derived
    `future vs frozen (no variant)` = b("ctc_future") - b("ctc_frozen"),
    `future vs frozen (variant)` = b("ctc_future") + b("ctc_future_variant") - b("ctc_frozen") - b("ctc_frozen_variant"),
    `variant | future contact` = b("variant") + b("ctc_future_variant"),
    `variant | frozen contact` = b("variant") + b("ctc_frozen_variant"),
    `nbhd | duration` = b("nbhd") + b("duration_nbhd"),
    `duration | nbhd` = b("duration") + b("duration_nbhd")
  )
  stopifnot(!anyNA(out))
  out
}
DERIVED_TERMS <- c("future vs frozen (no variant)", "future vs frozen (variant)", "variant | future contact",
                   "variant | frozen contact", "nbhd | duration", "duration | nbhd")

summarise_effects <- function(draws, family, horizon) {
  long <- melt(copy(draws)[, .draw := .I], id.vars = ".draw", variable.name = "term", value.name = "b")
  long[, .(ratio_median = exp(median(b)), ratio_lo = exp(quantile(b, 0.05)), ratio_hi = exp(quantile(b, 0.95)),
           pr_lt1 = mean(b < 0)), by = term][, `:=`(family = family, horizon = horizon)][]
}

# b_term + u_o per origin, the effect of `term` at each origin, on the multiplicative scale.
origin_slope_frame <- function(dr, fit, term) {
  lev <- levels(fit$model$origin)
  cols <- sprintf("s(origin):%s.%d", term, seq_along(lev))
  stopifnot(length(lev) == 55, all(cols %in% colnames(dr)))
  eff <- dr[, cols] + dr[, term]
  data.table(term = term, origin = as.Date(lev), ratio_median = exp(apply(eff, 2, median)),
             ratio_lo = exp(apply(eff, 2, quantile, 0.05)), ratio_hi = exp(apply(eff, 2, quantile, 0.95)))
}

# Random intercepts of one grouping term (`s(age_group)`, `s(origin)`), on the multiplicative scale.
re_level_frame <- function(dr, fit, var) {
  lev <- levels(fit$model[[var]])
  cols <- sprintf("s(%s).%d", var, seq_along(lev))
  stopifnot(all(cols %in% colnames(dr)))
  data.table(level = lev, ratio_median = exp(apply(dr[, cols], 2, median)),
             ratio_lo = exp(apply(dr[, cols], 2, quantile, 0.05)), ratio_hi = exp(apply(dr[, cols], 2, quantile, 0.95)))
}

# SD (log scale) and 95% interval of each random-effect term.
re_sd_table <- function(fit) {
  m <- gam.vcomp(fit, conf.lev = 0.95)$vc   # rows = re terms; `$all` is NULL for these models
  stopifnot(!is.null(m), all(c("std.dev", "lower", "upper") %in% colnames(m)))
  data.table(term = rownames(m), sd = m[, "std.dev"], sd_lo = m[, "lower"], sd_hi = m[, "upper"])
}

# Simulated WIS replicates from the fitted means and the estimated scale (no parameter uncertainty).
simulate_wis <- function(fit, fam, nsim, seed = 1236) {
  set.seed(seed)
  mu <- fitted(fit); n <- length(mu); phi <- fit$sig2   # NOT `fit$scale`: that partial-matches `scale.estimated`
  t(replicate(nsim, switch(fam,
    lognormal = exp(rnorm(n, mu, sqrt(phi))),
    gamma     = rgamma(n, shape = 1 / phi, scale = mu * phi),
    tweedie   = rtweedie(n, mu, phi, tweedie_p(fit)))))
}

# 9j's `PERIODS` (src/9j_viz_utils.jl), restated for the over-time figures. Keep in step by hand.
PERIODS <- data.table(
  period = c("Tiers", "Lockdown 2", "Lockdown 2 Easing", "Christmas", "Lockdown 3", "Lockdown 3 Schools open",
             "Lockdown 3 Easing", "Opening up", "Plan B / Omicron"),
  start = as.Date(c("2020-10-14", "2020-11-05", "2020-12-03", "2020-12-20", "2021-01-05", "2021-03-09",
                    "2021-03-29", "2021-10-01", "2021-11-25")),
  end   = as.Date(c("2020-11-04", "2020-12-02", "2020-12-19", "2021-01-04", "2021-03-08", "2021-03-28",
                    "2021-09-30", "2021-11-24", "2022-01-26"))
)
