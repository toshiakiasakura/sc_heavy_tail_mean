# Visualisation helpers shared between aggregated and raw-scatter panels in
# 2j_proportion_duration_physical.ipynb. Builds proportion-vs-degree panels
# with a log-scaled x-axis, log10 marker sizing, and per-panel xtick truncation.

# Category labels shown in legends/titles.
category_names_viz = Dict(
    :duration_multi    => ["<5min", "5–15min", "15min–1hr", "1–4hr", "4+hr"],
    :duration_multi_na => ["<5min", "5–15min", "15min–1hr", "1–4hr", "4+hr", "NA"],
    :phys_contact      => ["physical", "non-physical"],
    :duration_danon    => ["<10min", "11–30min", "31–60min", ">60min"])

# Candidate raw-number ticks for log-scaled degree axes.
_xticks_raw = [1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000]

# Per-panel xticks: drop ticks past the panel's max degree (with 1.1 headroom),
# so a small-setting (home) panel doesn't borrow a non-home panel's wide ticks.
function _xticks_for_max(max_n::Real)
    ticks = filter(x -> x <= max_n * 1.1, _xticks_raw)
    isempty(ticks) && (ticks = [1])
    return (ticks, string.(ticks))
end

# Marker size on a log10 scale; +1 inside the log avoids a zero size at count=1.
_marker_size_log(counts; base = 1.2, scale = 1.0) = log10.(counts .+ 1) .* scale .+ base

function _emp_props_by_n(Y::Matrix{Int}, n::Vector{Int}, K::Int)
    df_e = DataFrame(n = n)
    for k in 1:K; df_e[!, Symbol("y$k")] = Y[:, k]; end
    g = combine(groupby(df_e, :n)) do sub
        nobs = sum(sub[:, :n])
        (; (Symbol("p$k") => sum(sub[:, Symbol("y$k")]) / nobs for k in 1:K)...,
           ncell = nrow(sub), nobs = nobs)
    end
    return sort!(g, :n)
end

# Like `_emp_props_by_n`, but groups by `n_grouping` (e.g. degree-incl-NA) and
# divides by an independent per-cell denominator (e.g. non-NA contact count).
# The x-axis reflects the full degree while proportions are computed only over
# the contacts whose category was actually observed.
function _emp_props_by_n_with_denom(Y::Matrix{Int}, n_grouping::Vector{Int},
                                    denom::Vector{Int}, K::Int)
    df_e = DataFrame(n = n_grouping, denom = denom)
    for k in 1:K; df_e[!, Symbol("y$k")] = Y[:, k]; end
    g = combine(groupby(df_e, :n)) do sub
        d    = sum(sub[:, :denom])
        nobs = sum(sub[:, :n])
        (; (Symbol("p$k") => (d > 0 ? sum(sub[:, Symbol("y$k")]) / d : 0.0)
            for k in 1:K)...,
           ncell = nrow(sub), nobs = nobs)
    end
    return sort!(g, :n)
end

# DM-inputs builder that keeps NA as its own (last) duration category.
# Mirrors `prepare_dm_inputs` but maps NA → category K instead of imputing or dropping.
function _prepare_duration_inputs_na_cat(df_in::DataFrame; setting::String)
    is_missing_val(v) = ismissing(v) || (v isa AbstractString && v == "NA")
    to_int(v) = v isa AbstractString ? parse(Int, v) : Int(v)

    K = 6
    df_s = setting == "home" ? @subset(df_in, :cnt_home .== "true") :
                                @subset(df_in, :cnt_home .== "false")
    df_s = copy(df_s)
    df_s[!, :duration_multi] = [is_missing_val(v) ? K : to_int(v)
                                for v in df_s[:, :duration_multi]]
    grp = combine(groupby(df_s, [:part_id_d, :date])) do sub
        counts = zeros(Int, K)
        for v in sub[:, :duration_multi]
            1 <= v <= K && (counts[v] += 1)
        end
        (; (Symbol("y$k") => counts[k] for k in 1:K)..., n = sum(counts))
    end
    grp = @subset(grp, :n .> 0)
    Y = Matrix{Int}(grp[:, [Symbol("y$k") for k in 1:K]])
    n = Vector{Int}(grp[:, :n])
    return (Y = Y, n = n, K = K)
end

# Drop NA contacts when computing proportions, but keep the degree (x-axis)
# including the NA contacts. Returns:
#   :n       — full per-cell contact count (incl NA-duration contacts)
#   :n_nonna — per-cell count of non-NA-duration contacts (= sum of Y)
#   :Y       — K=5 category counts (NA excluded)
function _prepare_duration_inputs_drop_na_keep_n(df_in::DataFrame; setting::String)
    is_missing_val(v) = ismissing(v) || (v isa AbstractString && v == "NA")
    to_int(v) = v isa AbstractString ? parse(Int, v) : Int(v)

    K = 5
    df_s = setting == "home" ? @subset(df_in, :cnt_home .== "true") :
                                @subset(df_in, :cnt_home .== "false")
    grp = combine(groupby(df_s, [:part_id_d, :date])) do sub
        counts  = zeros(Int, K)
        n_total = nrow(sub)
        for v in sub[:, :duration_multi]
            if !is_missing_val(v)
                kk = to_int(v)
                1 <= kk <= K && (counts[kk] += 1)
            end
        end
        (; (Symbol("y$k") => counts[k] for k in 1:K)...,
            n = n_total, n_nonna = sum(counts))
    end
    grp = @subset(grp, :n .> 0)
    Y       = Matrix{Int}(grp[:, [Symbol("y$k") for k in 1:K]])
    n       = Vector{Int}(grp[:, :n])
    n_nonna = Vector{Int}(grp[:, :n_nonna])
    return (Y = Y, n = n, n_nonna = n_nonna, K = K)
end

function _props_panel_base(emp::DataFrame, K::Int, names::Vector{String}; title::AbstractString)
    max_n = maximum(emp.n)
    p = plot(xlabel = "degree", ylabel = "proportion",
        title = title, legend = :topright,
        ylim = (-0.02, 1.02),
        xscale = :log10,
        xticks = _xticks_for_max(max_n),
        xlim = (0.9, max_n * 1.1))
    ms = _marker_size_log(emp.nobs)
    for k in 1:K
        scatter!(p, emp.n, emp[!, Symbol("p$k")];
            ms = ms, msw = 0, alpha = 0.7,
            color = palette(:default)[k], label = names[k])
    end
    return p
end

function _plot_props_panel(df_in::DataFrame, setting::AbstractString,
                           outcome::Symbol, K::Int;
                           title_suffix::AbstractString = "")
    inp = prepare_dm_inputs(df_in; setting = setting, outcome = outcome, K = K)
    emp = _emp_props_by_n(inp.Y, inp.n, K)
    return _props_panel_base(emp, K, category_names_viz[outcome];
        title = string(outcome, title_suffix, " — ", setting))
end

function _plot_props_panel_dur_na(df_in::DataFrame, setting::AbstractString)
    inp = _prepare_duration_inputs_na_cat(df_in; setting = setting)
    emp = _emp_props_by_n(inp.Y, inp.n, inp.K)
    return _props_panel_base(emp, inp.K, category_names_viz[:duration_multi_na];
        title = string("duration_multi (NA as category) — ", setting))
end

function _plot_props_panel_dur_dropna(df_in::DataFrame, setting::AbstractString)
    inp = _prepare_duration_inputs_drop_na_keep_n(df_in; setting = setting)
    emp = _emp_props_by_n_with_denom(inp.Y, inp.n, inp.n_nonna, inp.K)
    return _props_panel_base(emp, inp.K, category_names_viz[:duration_multi];
        title = string("duration_multi (NA excl from prop, deg incl NA) — ", setting))
end

# Raw per-cell scatter helpers (no degree-aggregation).
function _raw_proportion_points(inp::NamedTuple, k::Int)
    pts = DataFrame(n = inp.n, p = inp.Y[:, k] ./ inp.n)
    return combine(groupby(pts, [:n, :p]), nrow => :count)
end

function _build_raw_panel(pts::DataFrame, k::Int, title::AbstractString)
    max_n = maximum(pts.n)
    return scatter(pts.n, pts.p;
        ms = _marker_size_log(pts.count; scale = 2.0),
        msw = 0, alpha = 0.55,
        color = palette(:default)[k],
        xlabel = "degree", ylabel = "proportion",
        xscale = :log10,
        xticks = _xticks_for_max(max_n),
        xlim = (0.9, max_n * 1.1),
        ylim = (-0.02, 1.02),
        title = title,
        legend = false)
end

function _plot_raw_panel(df_in::DataFrame, setting::AbstractString,
                         outcome::Symbol, K::Int, k::Int;
                         names = category_names_viz[outcome])
    inp = prepare_dm_inputs(df_in; setting = setting, outcome = outcome, K = K)
    pts = _raw_proportion_points(inp, k)
    return _build_raw_panel(pts, k, string(names[k], " — ", setting))
end

function _plot_raw_panel_dur_na(df_in::DataFrame, setting::AbstractString, k::Int)
    inp = _prepare_duration_inputs_na_cat(df_in; setting = setting)
    pts = _raw_proportion_points(inp, k)
    return _build_raw_panel(pts, k,
        string(category_names_viz[:duration_multi_na][k], " — ", setting))
end
