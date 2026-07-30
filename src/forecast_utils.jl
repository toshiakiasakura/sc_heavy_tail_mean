# forecast_utils.jl — canonical preamble for the composable forecasting framework.
#
# A single `include("forecast_utils.jl")` from `src/` loads everything needed to
# build, fit, and score the age-pair contact-degree renewal/NGM model
# (see inst/1a_preliminary_framework_plan.md and inst/1b_8j_model_structure.md):
#
#   • the base project preamble (main_utils.jl) — all `using` statements, the custom
#     Distributions library, DegreeDist, and the Turing/Pathfinder stack;
#   • the CoMix-UK data pipeline that supplies the raw contacts
#     (data_setup.jl, comix_uk_time_series.jl);
#   • the forecasting-framework modules, in dependency order.
#
# After this include the full API of inst/1a/1b is in scope, notably:
#   FrameworkConfig, WeeklyWindow, cis_age_grid                         (framework.jl)
#   load_window_data, load_forecast_truth                              (infection_data.jl)
#   prepare_degree_data, pool_over_time                                (degree_agepair.jl)
#   gen_interval_logparams, gen_interval_pmf_log, gen_interval_pmf,
#   gi_moments_days, renewal_next, forecast_forward                    (renewal.jl)
#   MeanNGM / NeighbourhoodDegreeNGM / NullNGM / DiagonalMeanNGM,
#     build_ngm, contact_star                                          (ngm.jl)
#   NegBinAgePair / HurdleWeibullAgePair / NoContactDegree, model_degree, model_transmission,
#     fit_stage1, stage2_inputs, fit_stage2_pooled, two_stage_forecast, build_degree_stats,
#     null_contact_level / null_moment_draws,
#     prefit_stage1! / prefit_stage2! / prefit_two_stage!               (joint_model.jl)
#   to_quantile_long, score_wis, to_sample_long, score_logs,
#     crps_sample, mean_crps                                           (scoring.jl)
#
# Run scripts/notebooks from `src/` — relative data paths assume `cwd == src/`.
# `score_wis` additionally requires the R `scoringutils` package (install once with
# `install.packages("scoringutils")`).

# --- base preamble + CoMix data pipeline -------------------------------------
include("main_utils.jl")            # using-statements + distributions + degree_dist + turing_*
include("data_setup.jl")            # CoMix-UK raw loaders, read_arrow_df, _uk_duration_multi, ...
include("comix_uk_time_series.jl")  # inc2prev-aligned weekly helpers

# --- forecasting framework (dependency order) --------------------------------
include("framework.jl")             # swap-axis types, WeeklyWindow, FrameworkConfig, containers, age grid
include("infection_data.jl")        # weekly infections (rolling-sum × pop) + gen_dab antibody
include("degree_agepair.jl")        # age-pair binning + per-cell weekly degree assembly
include("renewal.jl")               # generation-interval PMF + renewal iteration/forecast
include("ngm.jl")                   # NGM builders (mean / neighbourhood) + reciprocity + susceptibility
include("joint_model.jl")           # two-stage (cut) models + Pathfinder→NUTS fits + pooled forecast
include("scoring.jl")               # WIS via scoringutils (RCall) + native CRPS cross-check
