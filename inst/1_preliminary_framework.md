I would want to extend the study of the following 
- Paper: `./inst/pcbi.1011453.pdf`
- Corresponding code: /Users/toshi/Desktop/prj_validate_studies/CovidAgeGroupForecast
- New analysis plan: `inst/analysis_plan_heavy_tail_mean.docx`

Firstly I would want you to implement the framework, 
and fitting process should be done only for the first week of 2021 (using 8 weeks length),
and calculating the forecasting scores in four ways for the preliminary analysis. 
Also, fitting should use PathFinder.jl for initial value findings and parsimonious fit, 
and Turing.jl for the formal fitting. 

Composable implementaition, particularly for contact degree fitting, and 
renewal equation parts, since those compoents will be added or swapped. 