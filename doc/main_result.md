
# Synopsis of content

- [Synopsis of content](#synopsis-of-content)
- [Main figures structure](#main-figures-structure)
  - [Age, period forecasting performancs](#age-period-forecasting-performancs)
  - [Model validity](#model-validity)
  - [Contact characteristics](#contact-characteristics)
- [Figure candidates](#figure-candidates)
  - [Fan charts](#fan-charts)
  - [One joint parameter visualisations](#one-joint-parameter-visualisations)
  - [Scoring parts](#scoring-parts)
  - [Modelling scores](#modelling-scores)
  - [Contact characteristics](#contact-characteristics-1)
  - [Data usage](#data-usage)
  - [Supplementary figures.](#supplementary-figures)
  - [Coverage and bias](#coverage-and-bias)


# Main figures structure

TODO: degree distribution difference. 


![Caption](../res/18j_fans_patterns_parameters_pre_omicron_column_fans.png)
Figure 2. Forecasting result every 4 weeks for observed, hweibull|nbhd contact only, hweibull|nbhd contact - variant, neigbin|mean contact -variant. 


![Caption](../res/18j_summary_pre_omicron.png)
Figure 4. Effects of each model component. 

##  Age, period forecasting performancs
![Caption](../res/18j_age_rel_wis_h1_h4.png)
Figure x. Relative WIS by age. 
Values used for colouring were capped at 1+-0.5.

![Caption](../res/17j_scenarios_wis_by_period_weighted-hweibull_neighbourhood.png)
Figure 3. Scores by period for hweibull|nbhd.

## Model validity 
Secondary attack rates, and infectivity and susceptiblity. 

![Caption](../res/18j_fans_patterns_parameters_pre_omicron_column_parameters.png)

## Contact characteristics
![Caption](../res/18j_contact_rate_reproduction_pre_omicron.png)




# Figure candidates
## Fan charts
![Caption](../res/18j_fans_over_time_negbin_pre_omicron.png)
Figure 2. Forecasting result every 4 weeks for observed, hweibull|nbhd contact only, hweibull|nbhd contact - variant, neigbin|mean contact -variant. 

![Cpation](../res/18j_fans_over_time_negbin_pre_omicron_linear.png)
Figure Sx. FFrecasting result in a linear scale. 

![Caption](../res/18j_fans_over_time_negbin_pre_omicron_median.png)

All should be used for the main figure. Patterns would be the followings:
Pattern 1 (hweibull|nbhd models, best without one data):
- contact - variant 
- contact - only
- no contact - variant 

Pattern 2 (various models, all contact + variant):
- hweibull|nbhd
- hweibull|mean
- negbin|nbhd
- negbin|mean

Pattern 3 (hweibull|nbhd, frozen comparison):
- contact - variant 
- contact (frozen) - variant
- contact - only
- contact (frozen) - only

![Caption](../res/18j_fans_patterns_pre_omicron.png)

## One joint parameter visualisations
Using hweibull|nbhd, contact - variant model. 

- Forecast funs (all age)
- gamma_SAR (with share of proportions)
- Age-dependent susceptibility. 
- Age-dependent infectivity. 
- Age-dependent vaccination coverage

![Caption](../res/18j_variant_summary_column_pre_omicron.png)

![Caption](../res/18j_fans_patterns_parameters_pre_omicron_column_fans.png)
![Caption](../res/18j_fans_patterns_parameters_pre_omicron_column_parameters.png)

![Caption](../res/18j_variant_transmission_pre_omicron.png)

## Scoring parts
![Caption](../res/18j_scenarios_dots_h1_h4.png)
![Caption](../res/18j_scenarios_dots_h1_h2_h3_h4.png)


![Caption](../res/18j_age_rel_wis_h1_h4.png)
Figure x. Relative WIS by age. 

![Caption](../res/17j_scenarios_wis_by_period_weighted-hweibull_neighbourhood.png)
Figure 3. Scores by period for hweibull|nbhd.

![Caption](../res/17j_scenarios_point_forecasts_age_2020-12-07_weighted-hweibull_neighbourhood.png)
FIgure x. How adding each model results in the outcome. 

## Modelling scores

Equation used for the regression. 

bam(wis ~ ctc_future + ctc_frozen + variant + ctc_future_variant + ctc_frozen_variant + 
          duration + nbhd + duration_nbhd + diagonal +  
          s(origin, bs = "re") + s(age_group, bs = "re") + s(origin_age, bs = "re") +  
          s(origin, by = ctc_future, bs = "re")         + s(origin, by = ctc_frozen, bs = "re") +  
          s(origin, by = variant, bs = "re")            + s(origin, by = ctc_future_variant, bs = "re") + 
          s(origin, by = ctc_frozen_variant, bs = "re") + s(origin, by = duration, bs = "re") +  
          s(origin, by = nbhd, bs = "re")               + s(origin, by = duration_nbhd, bs = "re") +  
          s(origin, by = diagonal, bs = "re"),  
    data = d_all[horizon == h],  
    family = tw(link = "log"),           # or Gamma(link = "log"); or gaussian() with log(wis) on the left 
    method = "fREML", discrete = TRUE, nthreads = parallel::detectCores() - 1)  

lme4 format can be written as
m <- lmer(  
  wis ~ ctc_future + ctc_frozen + variant + ctc_future_variant + ctc_frozen_variant +  
        duration + nbhd + duration_nbhd + diagonal +  
        (1 + ctc_future + ctc_frozen + variant + ctc_future_variant + ctc_frozen_variant +  
             duration + nbhd + duration_nbhd + diagonal || origin) +  
        (1 | age_group) +  
        (1 | origin:age_group),  
  data = dat
)

![Caption](../res/19r_fixed_effects_tweedie.png)
Figure 4. Effects of each model component. 

![Caption](../res/19r_origin_slopes.png)
Figure Sx. Random effect values over time. 

![Caption](../res/19r_ppc.png)
Figure Sx. Posterior predictive check for WIS scores. 

## Contact characteristics

![Caption](../res/14j_duration_share.png)
Figure Sx. Duration share by home and non-home by adult and child. 
This should be modified to represent an average duration, 
with those with at least one contact or not over time.  

![Caption](../res/14j_duration_minutes.png)
Figure Sx. Duration minuts. 


![Caption](../res/14j_contact_quantities_by_block.png)
Figure x. Mean and excess contact over time. 


![Caption](../res/9j_reproduction_number.png)
Figure Sx. Reproduction number derived from estimated next generation matrix. 

![Caption](../res/9j_contact_reproduction_number.png)
Figure Sx. Reproduction number derived from the contact matrices. 

## Data usage
![Caption](../res/17j_scenarios_point_forecasts_age_2020-12-07_weighted-hweibull_neighbourhood.png)
FIgure x. How adding each model results in the outcome. 

## Supplementary figures. 
![Caption](../res/1j_data_availability_england.png)

![Caption](../res/1j_outcome_infections_vs_positives_england.png)
Figure Sx. Inc2prev value comparison. 

![Caption](../res/1j_variant_mix_england.png)
COG UK data availability. 


Note: these results are comming from the contact only (without variant model).
![Caption](../res/8j_susc_ratio_bins_over_time.png)
Finest bins for susceptibility.
![Caption](../res/8j_susc_ratio_over_time.png)
Bins are rounded for susceptibility.

![Caption](../res/8j_infectivity_ratio_over_time.png)
Infectivity by grouped age bins.

![Caption](../res/10j_degdist_unweighted-negbin_mean_2021-05-10.png)
![Caption](../res/10j_degdist_weighted-hweibull_mean_2021-05-10.png)

## Coverage and bias 
![Caption](../res/17j_scenarios_coverage_50_90_pre_omicron.png)
Figure x. Coverage by models.
![Caption](../res/17j_scenarios_bias_by_horizon_pre_omicron.png)
Figure x. Bias by models.