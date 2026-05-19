

function read_comix_uk_adult_time_series()
    df = read_comix_uk_contact()  |> standardise_comix_uk_to_socialmixer_data;
    df_part = DataFrame(JDF.load("../dt_comix_no_public/part_uk.jdf";
        cols = [:part_wave_uid, :part_age_group, :part_gender_nb, :date]));

    df_part = @select(df_part,
        :part_id = :part_wave_uid,
        :part_age = :part_age_group,
        :part_gender = :part_gender_nb,
        :date
        )

    df_part = filter_adult_cate(df_part, col = :part_age)
    df_part_ch = add_date_chunks(df_part);
    df = leftjoin(df, df_part_ch, on = [:part_id]);
    return (df, df_part_ch)
end

function create_chunk_df(df_part::DataFrame; chunk_days::Int = 14, anchor_date::Union{Nothing,Date} = nothing)
    earliest_date = minimum(df_part[:, :date])
    latest_date   = maximum(df_part[:, :date])

    # Anchor: aligns chunk boundaries to a fixed date so multiple datasets
    # share the same week labels. The first emitted chunk is the anchor-aligned
    # one that contains `earliest_date`; subsequent chunks tile forward.
    start = anchor_date === nothing ? earliest_date : anchor_date
    Δ = (earliest_date - start).value
    if Δ >= 0
        offset = Δ ÷ chunk_days
        start  = start + Day(offset * chunk_days)
    else
        offset = ceil(Int, -Δ / chunk_days)
        start  = start - Day(offset * chunk_days)
    end

    chunks = []
    current = start
    chunk_num = 1
    while current <= latest_date
        chunk_end = current + Day(chunk_days - 1)
        mid_date  = current + Day(div(chunk_end - current, 2))
        push!(chunks, (chunk_number=chunk_num, chunk_start=current, chunk_end=chunk_end, mid_date=mid_date))
        current = chunk_end + Day(1)
        chunk_num += 1
    end

    return DataFrame(chunks)
end

"""
Anchor date for inc2prev-style 7-day weeks (Sunday-start), as used by
`/workdir/inc2prev/scripts/read.R` (`seq(as.Date("2021-03-21"), ..., by = 7)`).
"""
inc2prev_week_anchor() = Date(2021, 3, 21)

"""
Build inc2prev-aligned 7-day week labels covering `[start_date, end_date]`.
"""
function create_week_df(df_part::DataFrame; start_date::Date = Date(2021, 7, 1),
                        end_date::Date = Date(2021, 12, 31))
    df_filt = @subset(df_part, start_date .<= :date .<= end_date)
    return create_chunk_df(df_filt; chunk_days = 7, anchor_date = inc2prev_week_anchor())
end

function assign_chunk(date, df_chunk)
    df_tmp = @subset(df_chunk, :chunk_start .<= date .<= :chunk_end)
    if nrow(df_tmp) == 0
        error("Date $date does not fall into any chunk.")
    end
    return df_tmp[1, :chunk_number]
end

function add_date_chunks(df_part::DataFrame)
    df_chunk = create_chunk_df(df_part)
    df_part_chunked = transform(df_part,
        :date => ByRow(d -> assign_chunk(d, df_chunk)) => :chunk_number)
    df_part_chunked = leftjoin(df_part_chunked, df_chunk, on=:chunk_number)
    return df_part_chunked
end

function create_df_dds_chunk(df::DataFrame, df_part::DataFrame)
    d_lis = df_part[:, :mid_date] |> unique
    df_dd_mer = DataFrame()
    for d in d_lis
        df_tmp = @subset(df, :mid_date .== d)
        df_part_tmp = @subset(df_part, :mid_date .== d)
        df_dd, _ = get_df_dd_single(df_tmp, df_part_tmp, string(d))
        if nrow(df_dd) == 0
            println(d)
            continue
        end
        df_dd_mer = vcat(df_dd_mer, df_dd)
    end
    return df_dd_mer
end

"""
Variant of `create_df_dds_chunk` that builds degree distributions split by
contact location (`home`, `work`, `school`, `other`) using `degree_dist_by_location`.

`cnt_home`, `cnt_work`, `cnt_school` may be Bool or the string values
`"true"`/`"false"`/`"NA"` — both are handled. `"NA"` is treated as `false`.
"""
function create_df_dds_chunk_by_settings(df::DataFrame, df_part::DataFrame)
    to_bool(v) = string(v) == "true"

    d_lis = df_part[:, :mid_date] |> unique
    df_dd_mer = DataFrame()
    for d in sort(d_lis)
        df_tmp      = copy(@subset(df,      :mid_date .== d))
        df_part_tmp = copy(@subset(df_part, :mid_date .== d))
        if nrow(df_tmp) == 0 || nrow(df_part_tmp) == 0
            println("Skipping (no data): $d")
            continue
        end

        # degree_dist_by_location expects :part_id_d
        @rename! df_tmp      :part_id_d = :part_id
        @rename! df_part_tmp :part_id_d = :part_id

        # Normalise location columns to Bool ("NA" → false)
        df_tmp[!, :cnt_home]   = to_bool.(df_tmp[:, :cnt_home])
        df_tmp[!, :cnt_work]   = to_bool.(df_tmp[:, :cnt_work])
        df_tmp[!, :cnt_school] = to_bool.(df_tmp[:, :cnt_school])

        df_dd = degree_dist_by_location(df_tmp, df_part_tmp)
        df_dd[!, :key] .= string(d)
        df_dd_mer = vcat(df_dd_mer, df_dd)
    end
    return df_dd_mer
end

function create_capped_frequencies(df_k::DataFrame)
    df_k_capped = copy(df_k)
    df_below = @subset(df_k_capped, :x .<= 50)
    df_above = @subset(df_k_capped, :x .> 50)

    # Sum frequencies for degrees >50 and create a row with x=50
    if nrow(df_above) > 0
        freq_above_50 = sum(df_above.y)

        # Check if x=50 already exists in df_below
        if 50 in df_below.x
            # Add to existing x=50 frequency
            idx = findfirst(df_below.x .== 50)
            df_below.y[idx] += freq_above_50
        else
            # Create new row for x=50
            # Get a sample row to copy structure
            new_row = df_above[1, :]
            new_row.x = 50
            new_row.y = freq_above_50
            push!(df_below, new_row)
        end
    end
    return df_below
end

function get_mean_over_time(df_dds::DataFrame)
    results = []
    for k in unique(df_dds.key)
        for s in ["home", "non-home"]
            df_k = @subset(df_dds, :key .== k, :strat .== s)
            if nrow(df_k) > 0
                dds = DegreeDist(df_k)
                m = mean(dds)
                dds_cap = create_capped_frequencies(df_k) |> DegreeDist
                m_cap = mean(dds_cap)
                push!(results, (key=k, strat=s, mean=m, mean_cap=m_cap))
            end
        end
    end

    df_viz = DataFrame(results)
    sort!(df_viz, :key)
    return df_viz
end