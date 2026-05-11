######################################################################
###### Danon 2013 dataset loader and Danon-specific helpers  #########
######################################################################
#
# Reshapes `dt_Leon_Danon_2013/` into the project schema used by
# `prepare_dm_inputs`, `contact_degrees`, `fit_mglm_dm`, etc.:
#
#   df:      :part_id_d, :date, :cnt_home, :duration_multi (1..4 or "NA"),
#            :c_number, :c_main_where
#   df_part: :part_id_d, :date, :p_total_contacts, :p_age, :p_gender,
#            :p_household_size, :p_contact_hours
#
# `:date` is a constant placeholder (the Danon survey is single-shot per
# responder); it is included only because `prepare_dm_inputs` and
# `contact_degrees` key on `(part_id_d, date)`.
#
# Duration codes in the CSV are {-1, 0, 1, 2, 3} — offset by one from the
# codebook's documented {1..4}. We map 0/1/2/3 → project levels 1/2/3/4
# and -1 → "NA". Per `inst/4_Danon_analysis.md`, the weight reference is
# `d_max = 60 min` (so `w_{>60min} = 1`).

const _DANON_PATH              = joinpath(@__DIR__, "..", "dt_Leon_Danon_2013")
const _DANON_PLACEHOLDER_DATE  = Date(2010, 1, 1)

const _DANON_T_MID = (5.0, 20.0, 45.0, Inf)
const _DANON_LABELS = ("<10 min", "11–30 min", "31–60 min", ">60 min")

# Finite midpoint used when reinterpreting a >60-min bin under :per_person
# (Inf would force everything into level 4). 60 is the conservative choice
# consistent with `duration_weight_danon` (which caps level 4 at d_max=60).
const _DANON_T_MID_FINITE = (5.0, 20.0, 45.0, 60.0)

"""
    read_danon_contacts(; data_dir = _DANON_PATH,
                          disaggregate = true,
                          duration_mode = :as_is)

Load `Contact_data.csv` + `Person_data.csv` and reshape into the project
schema. Drops the 43 contact rows with `C_Number ∈ {-1, 0}`. Maps Danon
codes 0/1/2/3 → project levels 1/2/3/4 with `-1 → "NA"`. Sets
`cnt_home = "true"` when `C_Wheres_1 == 1`, else `"false"`.

When `disaggregate=true` (default), each contact row with `C_Number = k`
is expanded into `k` individual-contact rows so that one row = one
person. `duration_mode` controls how the recorded duration is propagated
to the disaggregated rows:

  - `:as_is` (default): every disaggregated copy keeps the original
    `duration_multi`. Co-presence interpretation — the duration is the
    time co-present with each member of the group.
  - `:per_person`: rebin the duration as `t_mid / k` and remap to the
    nearest Danon level (NA preserved). Sensitivity for the known
    misreporting pattern where respondents enter the *total group event
    time* rather than the per-individual contact time. Level-4 midpoint
    is taken as 60 min (the conservative lower-bound finite value).

Returns `(df, df_part)`.
"""
function read_danon_contacts(; data_dir = _DANON_PATH,
                               disaggregate::Bool = true,
                               duration_mode::Symbol = :as_is)
    df_c = CSV.read(joinpath(data_dir, "Contact_data.csv"), DataFrame)
    df_p = CSV.read(joinpath(data_dir, "Person_data.csv"),  DataFrame)

    df_c = @subset(df_c, :C_Number .>= 1)

    df = DataFrame(
        part_id_d      = string.(df_c.C_PID),
        date           = fill(_DANON_PLACEHOLDER_DATE, nrow(df_c)),
        cnt_home       = [w == 1 ? "true" : "false" for w in df_c.C_Wheres_1],
        duration_multi = [d == -1 ? "NA" : string(d + 1) for d in df_c.C_Duration],
        c_number       = df_c.C_Number,
        c_main_where   = df_c.C_MainWhere,
    )

    df_part = DataFrame(
        part_id_d        = string.(df_p.P_ID),
        date             = fill(_DANON_PLACEHOLDER_DATE, nrow(df_p)),
        p_total_contacts = df_p.P_total_contacts,
        p_age            = df_p.P_age,
        p_gender         = df_p.P_gender,
        p_household_size = df_p.P_household_size,
        p_contact_hours  = df_p.P_Contact_Hours,
    )

    if disaggregate
        df = disaggregate_group_contacts(df; duration_mode = duration_mode)
    end

    return df, df_part
end

"""
    disaggregate_group_contacts(df; duration_mode = :as_is)

Expand each contact row in the reshape-schema `df` (columns
`:part_id_d, :date, :cnt_home, :duration_multi, :c_number, :c_main_where`)
into `c_number` individual-contact rows. `c_number` is preserved on
each disaggregated row (so the original group size remains queryable),
but the row is now the unit of one individual contact.

`duration_mode = :as_is` keeps the original `duration_multi`; `:per_person`
rebins by `t_mid / c_number` and remaps to the nearest Danon level (NA
stays NA; level-4 midpoint treated as 60 min).
"""
function disaggregate_group_contacts(df::DataFrame; duration_mode::Symbol = :as_is)
    duration_mode in (:as_is, :per_person) ||
        error("duration_mode must be :as_is or :per_person")

    counts = df.c_number
    any(counts .< 1) && error("disaggregate expects c_number .>= 1 (filter upstream)")

    new_dur = duration_mode == :as_is ? df.duration_multi :
              [_rebin_per_person(df.duration_multi[i], counts[i]) for i in 1:nrow(df)]

    idx = reduce(vcat, [fill(i, counts[i]) for i in 1:nrow(df)])
    out = df[idx, :]
    out.duration_multi = new_dur[idx]
    return out
end

# Map (level, group size) → new Danon level under per-person reinterpretation.
function _rebin_per_person(d, k::Integer)
    k <= 1            && return d
    _is_dur_na(d)     && return d
    d_int = _dur_to_int(d)
    (1 <= d_int <= 4) || return d
    t_per = _DANON_T_MID_FINITE[d_int] / k
    new_level = t_per <= 10 ? 1 :
                t_per <= 30 ? 2 :
                t_per <= 60 ? 3 : 4
    return string(new_level)
end

"""
    duration_weight_danon(d, d_max=60)

Per-contact weight on the Danon K=4 duration scale. `d ∈ {1..4}` or `"NA"`
(missing). NA is treated as level 1 (<10 min). Reference midpoints (min):
1→5, 2→20, 3→45, 4→cap at `d_max` (>60 min ⇒ w=1).

The closed form is `w = t_mid / d_max`, with `t_mid` clipped at `d_max`
for level 4. Requires `d_max ≥ 45`.
"""
function duration_weight_danon(d, d_max::Real = 60)
    d_eff = _is_dur_na(d) ? 1 : _dur_to_int(d)
    1 <= d_eff <= 4 || error("unexpected Danon :duration_multi value $d")
    t = min(_DANON_T_MID[d_eff], float(d_max))
    return t / d_max
end

"""
    print_duration_weights_danon(d_max; io=stdout)

Print `duration_weight_danon(d, d_max)` for each level `d ∈ 1..4`.
"""
function print_duration_weights_danon(d_max::Real; io::IO = stdout)
    println(io, "duration_weight_danon(d, d_max = $d_max min):")
    for d in 1:4
        t_mid = isfinite(_DANON_T_MID[d]) ? _DANON_T_MID[d] : float(d_max)
        @printf(io, "  d=%d  %-11s  t_mid=%5.1f min  w=%.6f\n",
                d, _DANON_LABELS[d], t_mid, duration_weight_danon(d, d_max))
    end
end

"""
    contact_degrees_danon(df, df_part; setting, weighted=false, d_max=60)

Per-(part_id_d, date) degree vector on the Danon dataset. Parallel to
`contact_degrees` but uses `duration_weight_danon` (K=4 scale).
"""
function contact_degrees_danon(df::DataFrame, df_part::DataFrame;
                               setting::Symbol, weighted::Bool = false,
                               d_max::Real = 60)
    sub = setting === :home    ? @subset(df, :cnt_home .== "true")  :
          setting === :nonhome ? @subset(df, :cnt_home .== "false") :
          setting === :all     ? df :
          error("setting must be :all, :home, or :nonhome")

    keys_part = unique(@select(df_part, :part_id_d, :date))

    if weighted
        sub = @transform(sub, :w = duration_weight_danon.(:duration_multi, d_max))
        agg = combine(groupby(sub, [:part_id_d, :date]), :w => sum => :deg)
        joined = leftjoin(keys_part, agg, on = [:part_id_d, :date])
        return coalesce.(joined.deg, 0.0)
    else
        agg = combine(groupby(sub, [:part_id_d, :date]), nrow => :deg)
        joined = leftjoin(keys_part, agg, on = [:part_id_d, :date])
        return coalesce.(joined.deg, 0)
    end
end

"""
    dm_expected_weight_danon(fit, n, d_max)

E[w | n] for the Danon K=4 scale under an MGLM Dirichlet-multinomial fit.
"""
function dm_expected_weight_danon(fit::NamedTuple, n::Integer, d_max::Real)
    n == 0 && return 0.0
    P = size(fit.β, 1)
    Xn = P == 1 ? reshape([1.0], 1, 1) : reshape([1.0, log(Float64(n))], 1, 2)
    p = mglm_dm_proportions(fit.β, Xn)
    w_levels = (duration_weight_danon(k, d_max) for k in 1:4)
    return sum(p[1, k] * w for (k, w) in enumerate(w_levels))
end

"""
    contact_degrees_dm_imputed_danon(df, df_part, fit; setting, d_max=60)

Danon analogue of `contact_degrees_dm_imputed`. NA-duration contacts are
imputed softly by `dm_expected_weight_danon(fit, n, d_max)` where `n` is
the participant's total contact count in the chosen setting (NA included).
`:all` is not supported — combine `:home + :nonhome` instead.
"""
function contact_degrees_dm_imputed_danon(df::DataFrame, df_part::DataFrame,
                                          fit::NamedTuple;
                                          setting::Symbol, d_max::Real = 60)
    sub = setting === :home    ? @subset(df, :cnt_home .== "true")  :
          setting === :nonhome ? @subset(df, :cnt_home .== "false") :
          error(":all is not supported here; combine :home + :nonhome instead")

    sub = @transform(sub,
        :w_obs = ifelse.(_is_dur_na.(:duration_multi),
                         0.0,
                         duration_weight_danon.(:duration_multi, d_max)),
        :is_na = _is_dur_na.(:duration_multi))

    agg = combine(groupby(sub, [:part_id_d, :date]),
                  :w_obs => sum => :w_sum,
                  :is_na => sum => :n_na,
                  nrow         => :n_tot)

    agg = @transform(agg,
        :deg = :w_sum .+ :n_na .* dm_expected_weight_danon.(Ref(fit), :n_tot, d_max))

    keys_part = unique(@select(df_part, :part_id_d, :date))
    joined = leftjoin(keys_part, @select(agg, :part_id_d, :date, :deg),
                      on = [:part_id_d, :date])
    return coalesce.(joined.deg, 0.0)
end
