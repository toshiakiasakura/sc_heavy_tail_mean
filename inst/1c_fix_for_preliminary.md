For the 8j implementation, the following should be fixed. 

The neighbourhood-degree NGM should be calculated
based on those with more than 0 degress/durations. 

Therefore, first calculate the excess degree 
among non-zero values (i.e. for negative binomial, left-truncat at 0
and calculate the mean, for weighted weibull, just use the mean of weighted 
weibull and multiply the 1 - proportion of zero for each grid.)

Also, for the negative binomial, introduce the zero-inflated term.

Also, for the forecasting, we assume only the contact matrix is available 
at the time of h-1, and re-estimate everything at each iteration. 

Therefore, MCMC results should be saved locally for each time t 
and each horizon. 

Also, do a forecasting for 4 data points (4 weeks), forward from the current baseline. 