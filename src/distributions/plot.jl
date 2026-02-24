##### Plot functions #####
function plot_pdf(d::UnivariateDistribution; kwds...)
	k = [(1:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10000]
	pl = plot(xaxis = :log10, xlabel = "log10(k)", ylabel = "log10(pdf(k))", yaxis = [-30, 0])
	plot!(pl, k, log10.(pdf.(d, k)); kwds...)
end

function plot_pdf!(pl::Plots.Plot, d::UnivariateDistribution; kwds...)
	k = [(1:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10000]
	plot!(pl, k, log10.(pdf.(d, k)); kwds...)
end

function plot_pdf_raw!(pl::Plots.Plot, d::UnivariateDistribution; kwds...)
	k = [(1:20)...]
	#pl = plot(xlabel="k", ylabel="pdf(k)")
	plot!(pl, k, pdf.(d, k); kwds...)
end

function plot_ccdf!(pl::Plots.Plot, d::UnivariateDistribution; kwds...)
	k = [(1:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10_000]
	[ccdf.(d, k[end:-1:begin])] # To avoid the repetition errors. Create cache.
	plot!(pl, k, log10.(ccdf.(d, k)); xaxis = :log10, kwds...)
end

##### ZeroInfDist and ZeroTruncDist #####
function plot_ccdf!(pl::Plots.Plot, d::ZeroInfDist; kwds...)
	d_trunc = convert_ZeroInf_to_ZeroTrunc(d)
	plot_ccdf!(pl, d_trunc; kwds...)
end

function plot_ccdf!(pl::Plots.Plot, d::ZeroInfConvolutedDist; kwds...)
	d_trunc = ZeroTruncConvolutedDist(conv_d=d)
	plot_ccdf!(pl, d_trunc; kwds...)
end

function plot_pdf!(pl::Plots.Plot, d::UnivariateDistribution;
		conv_log10 = true, kwds...)
	k = [(0:9)..., (10:10:90)..., (100:100:900)..., (1000:1000:9000)..., 10000]
	y = pdf.(d, k)
	if conv_log10 == true
		y = log10.(y)
	end
	plot!(pl, k, y; kwds...)
end