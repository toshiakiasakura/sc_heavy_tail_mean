###################################
##### Shared helper functions #####
###################################

DIR_SURVEY = "../dt_surveys/"

"""
You can merge `comix_uk_sday.csv` to `contact_common.csv` with a merge key of `part_id`.
Use `read_raw_sc_data_with_sday` to do this automatically.
"""
function read_raw_sc_data(survey_name:: String)
	path_contact_common = string(DIR_SURVEY, survey_name, "_contact_common.csv")
	path_part_common = string(DIR_SURVEY, survey_name, "_participant_common.csv")
	df = CSV.read(path_contact_common, DataFrame)
	df_part = CSV.read(path_part_common, DataFrame)
	return (df, df_part)
end

"""
Read contact and participant data for `survey_name`, then left-join the
survey-day metadata from `<survey_name>_sday.csv` onto the contact table
using `part_id` as the merge key. Adds columns: `sday_id`, `dayofweek`,
`day`, `month`, `year`, `wave`.
"""
function read_raw_sc_data_with_sday(survey_name::String)
	df, df_part = read_raw_sc_data(survey_name)
	path_sday = string(DIR_SURVEY, survey_name, "_sday.csv")
	df_sday = CSV.read(path_sday, DataFrame)
	df_sday = @select(df_sday, :part_id, :sday_id, :dayofweek, :day, :month, :year, :wave)
	@transform!(df_sday, :date = Date.(:sday_id, dateformat"yyyy.mm.dd"))
	df_part = leftjoin(df_part, df_sday, on = :part_id)
	return (df, df_part)
end

function read_adult_chunks(df, df_part)
    df_part = @select(df_part,
        :part_id, :part_age, :part_gender, :date)
	df_part = filter_adult_cate(df_part, col = :part_age)
	df_part_ch = add_date_chunks(df_part);
	df = leftjoin(df, df_part_ch, on = :part_id)
	return df, df_part_ch
end

function filter_adult_cont_and_innerjoin(df::DataFrame, df_part::DataFrame; col = :part_age)
	if isa(df_part[1, :part_age], AbstractString) == true
		df_part = @subset(df_part, :part_age .!= "NA")
		df_part[!, :part_age] .= parse.(Int, df_part[:, :part_age])
	end

	df_part = @subset(df_part, :part_age .>= 18);
	df = innerjoin(df, df_part, on = :part_id);
	return df, df_part
end

"""Prepared for the CoMix UK data, and CoMix 2 data.
"""
function filter_adult_cate(df_part::DataFrame; col = :part_age_group)
	adult_age = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-120",
		"18-19", "20-24", "25-34", "35-44", "45-54", "55-64", "70-74",
		"25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
		"60-64", "65-69", "75-79", "80+",
	]
	return @subset(df_part, in.($col, Ref(adult_age)))
end

"""Remove contacts and part id from those with cnt_home is NA.
"""
function remove_cnt_home_na(df::DataFrame, df_part::DataFrame)
	cond = in.(df[:, :cnt_home], Ref(["true", "false"]))
	df_clean = df[cond, :]
	ids_rem = df[.~cond, :part_id_d] |> unique
	println("Number of removing ids: ", length(ids_rem))
	ids_part = df_part[:, :part_id_d]
	ids_incl = [id for id in ids_part if (id in ids_rem) == false]
	df_part_clean = @subset(df_part, in.(:part_id_d, Ref(ids_incl)))
	return df_clean, df_part_clean
end

"""This function is prepared to rename part_id to part_uid.
Note:
part_uid is prepared to distinguish 1:n part and count data relationships.
"""
function rename_part_id_to_uid!(df, df_part)
	if "part_id_d" in names(df)
		println("part_uid already in df")
		return nothing
	end
	@rename! df :part_id_d = :part_id
	@rename! df_part :part_id_d = :part_id
	return nothing
end

"""Standardise cnt_home answers and remove other answers.
"""
function standardise_cnt_home_values!(df::DataFrame)
	df[!, :cnt_home] = string.(df[:, :cnt_home])
	@transform! df :cnt_home = replace.(:cnt_home,
		Dict("TRUE" => "true", "FALSE" => "false",
			"1" => "true", "0" => "false",
		)...,
	)
end

function degree_dist_for_all_home_non_home(df::DataFrame; key=:key)
	df_all = @pipe combine(@groupby(df, [key, :part_id_d]), nrow => :cnt) |>
				   @transform(_, :strat = "all")
	# Note: this removed missing in cnt_home.
	if Set(df[:, :cnt_home]) != Set(["true", "false"])
		println("cnt_home contains missing")
	end
	df_hm = @pipe @subset(df, :cnt_home .== "true") |>
				  combine(@groupby(_, [key, :part_id_d]), nrow => :cnt) |>
				  @transform(_, :strat = "home")
	df_nhm = @pipe @subset(df, :cnt_home .== "false") |>
				   combine(@groupby(_, [key, :part_id_d]), nrow => :cnt) |>
				   @transform(_, :strat = "non-home")
	return vcat(df_all, df_hm, df_nhm)
end

function add_zero_counts_to_df_deg(df_deg::DataFrame, df_part::DataFrame)
	n_part = df_part[:, :part_id_d] |> unique |> length

	df_dd = vcat([
		@pipe @subset(df_deg, :strat .== st)[:, :cnt] |>
			  DegreeDist(_, n_part) |>
			  dd_to_df(_, st)
		for st in ["all", "home", "non-home"]
	]...)
	return df_dd
end

"""Clean standardised contact and part line data to DegreeDist style of DataFrame.
Args:
- df: contact line dataframe, which should be standardised and cleaned,
	which should only have single day answer (part_id is unique.)
- df_part: participant line dataframe, which should be standardised and cleaned.
"""
function get_df_dd_single(df::DataFrame, df_part::DataFrame, key::String)
	standardise_cnt_home_values!(df)
	rename_part_id_to_uid!(df, df_part)
	df, df_part = remove_cnt_home_na(df, df_part)
	df_dd, df_part = create_df_dd(df, df_part, key)
	return (df_dd, df_part)
end

"""
Args:
- df: part_id_d (unique for each contact answer) is included.
"""
function create_df_dd(df::DataFrame, df_part::DataFrame, key::String)
	df[:, :key] .= key
	df_deg = degree_dist_for_all_home_non_home(df)
	df_dd = add_zero_counts_to_df_deg(df_deg, df_part) # for no group contacts
	df_dd[:, :key] .= key
	df_part[:, :key] .= key
	return (df_dd, df_part)
end

"""For dataframe with two-day answers
"""
function duplicate_df_part(df_part::DataFrame)
	df_tmp1 = copy(df_part)
	df_tmp2 = copy(df_part)
	@transform!(df_tmp1, :part_id_d = string.(:part_id) .* "_1")
	@transform!(df_tmp2, :part_id_d = string.(:part_id) .* "_2")
	df_part_new = vcat(df_tmp1, df_tmp2);
	return df_part_new
end

function create_part_id_d_middle_number(df::DataFrame, df_part::DataFrame)
	df_part = duplicate_df_part(df_part)
	df[!, :part_id_d] = @. string(df[:, :part_id], "_", get_middle_number(df[:, :cont_id]))
	return (df, df_part)
end

function create_part_id_d_studyDay(df::DataFrame, df_part::DataFrame, r_survey)
	df_part = duplicate_df_part(df_part)
	df_cnt_extra = CSV.read(string(DIR_SURVEY, r_survey.file_contact_extra), DataFrame)
	df = leftjoin(df, df_cnt_extra, on = :cont_id);
	df[!, :part_id_d] = @with df string.(:part_id, "_", :studyDay)
	return (df, df_part)
end

"""
Function to get the middle number from a formatted string
e.g. 10_2_30, 10_1_29, 10_3_31

"""
function get_middle_number(s::AbstractString)
	parts = split(s, '_')  # Split the string on underscore
	if length(parts) == 3
		return parts[2]  # Parse the middle part to an integer and return
	elseif length(parts) == 4
		return parts[3]
	else
		return "NA"
	end
end

#################################################################
##### Social mixer standardised data with one single answer #####
#################################################################
"""
Note:
The eligible survey for this function is
- Only 1 day per participant.
- part_age is continuous.
- dataset is standardised for socialmixr.
"""
function read_dd_single_survey(df_master::DataFrame, key::String)
	r_survey = @subset(df_master, :key .== key)[1, :]
	df, df_part = read_raw_sc_data(r_survey);
	df, df_part = filter_adult_cont_and_innerjoin(df, df_part)
	df_dd, df_part = get_df_dd_single(df, df_part, key);
	return (df_dd, df_part)
end

function read_dd_two_day_survey(df_master::DataFrame, key::String)
	r_survey = @subset(df_master, :key .== key)[1, :]
	df, df_part = read_raw_sc_data(r_survey);
	df, df_part = filter_adult_cont_and_innerjoin(df, df_part);
	standardise_cnt_home_values!(df);

	if key in ["Melegaro_2017"]
		df, df_part = create_part_id_d_studyDay(df, df_part, r_survey);
	elseif key in ["Hens_2009", "Beraud_2015"]
		df, df_part = create_part_id_d_middle_number(df, df_part);
	else
		error("Key not recognised for two-day survey: ", key)
	end
	df, df_part = remove_cnt_home_na(df, df_part)

	df_dd, df_part = create_df_dd(df, df_part, key)
	return (df_dd, df_part)
end

#################################
##### Jonathan M. Read 2014 #####
#################################

function read_Read_2014_dds()
	path = "../dt_surveys/2014_Read_China.csv"
	df = CSV.read(path, DataFrame)
	df_tmp = filter_adult_cate(df; col=:age);
	df_all = df_tmp[:, Symbol("c.all")] |> DegreeDist |> dd_to_df
	df_all[:, :strat] .= "all"
	df_hm = df_tmp[:, Symbol("c.home")] |> DegreeDist |> dd_to_df
	df_hm[:, :strat] .= "home"
	df_nhm = (df_tmp[:, Symbol("c.all")] .- df_tmp[:, Symbol("c.home")]) |> DegreeDist |> dd_to_df
	df_nhm[:, :strat] .= "non-home"
	df_dd = vcat(df_all, df_hm, df_nhm)
	df_dd[:, :key] .= "Read_2014"
	return (df_dd, nothing)
end

#########################
##### Leung_2017    #####
#########################
# Non-stratified dds can be obtained by `read_dd_single_survey`.

"""
Note:
mode_survey = 1 is online, mode_survey = 2 is paper.
"""
function read_Leung_2017_paper_online()
	key = "Leung_2017"
	df_master = read_survey_master_data();
	r_survey = @subset(df_master, :key .== key)[1, :]

	df, df_part = read_raw_sc_data(r_survey);
	df_part_extra = CSV.read(string(DIR_SURVEY, r_survey.file_part_extra), DataFrame)
	df_part = leftjoin(df_part, df_part_extra, on = :part_id)

	df_part_paper = @subset(df_part, :mode_survey .== 1)
	df_tmp, df_part_paper = filter_adult_cont_and_innerjoin(copy(df), df_part_paper)
	df_dd_paper, df_part_paper = get_df_dd_single(df_tmp, df_part_paper, "Leung_2017_paper");

	df_part_online = @subset(df_part, :mode_survey .== 2)
	df_tmp, df_part_online = filter_adult_cont_and_innerjoin(copy(df), df_part_online)
	df_dd_online, df_part_online = get_df_dd_single(df_tmp, df_part_online, "Leung_2017_online");


	df_dd = vcat(df_dd_online, df_dd_paper)
	df_part = vcat(df_part_online, df_part_paper)
	return (df_dd, df_part)
end

#########################
##### Zhang_2019    #####
#########################

"""
Args:
- fil_survey_mode: takes false (all), "S" (Self-reporting), "T" (Telephone interview)
"""
function read_Zhang_2019_dds(; fil_survey_mode = false)
	key = "Zhang_2019"
	df_master = read_survey_master_data();
	r_survey = @subset(df_master, :key .== key)[1, :]

	df, df_part = read_raw_sc_data(r_survey);
	if isa(fil_survey_mode, String) == true
		df_part_extra = @pipe CSV.read(string(DIR_SURVEY, r_survey.file_part_extra), DataFrame)
		df_part = leftjoin(df_part, df_part_extra, on = :part_id)
		df_part = @subset(df_part, :mode_survey .== fil_survey_mode)
	end
	df, df_part = filter_adult_cont_and_innerjoin(df, df_part)
	standardise_cnt_home_values!(df)
	rename_part_id_to_uid!(df, df_part)
	df, df_part = remove_cnt_home_na(df, df_part)

	df[:, :key] .= key
	df_deg = degree_dist_for_all_home_non_home(df)
	df_deg = add_group_counts_to_Zhang_2019(df_deg, r_survey)
	df_dd = add_zero_counts_to_df_deg(df_deg, df_part) # for no group contacts
	df_dd[:, :key] .= key
	df_part[:, :key] .= key
	return (df_dd, df_part)
end

function add_group_counts_to_Zhang_2019(df_deg::DataFrame, r_survey)
	df_part_extra = @pipe CSV.read(string(DIR_SURVEY, r_survey.file_part_extra), DataFrame)
	df_part_extra = @select(df_part_extra, :part_id_d = :part_id, :group_n)

	df_deg_tmp = leftjoin(df_deg, df_part_extra, on = :part_id_d)
	@transform!(df_deg_tmp, @byrow :group_n = :strat != "home" ? :group_n : 0)
	@transform!(df_deg_tmp, :cnt = :cnt .+ :group_n)
	df_deg_tmp = @select(df_deg_tmp, Not(:group_n))
	return df_deg_tmp
end

#########################
##### Danon 2013     #####
#########################

function read_Danon_2013_dds()
	df, df_part = read_danon_2013_contact(; fil_adult = true);
	df_tmp = danon_degree_dist_for_all_hm_nhm(df, df_part)
	df_dd = add_zero_counts_to_df_deg(df_tmp, df_part)
	df_dd[:, :key] .= "Danon_2013"
	return (df_dd, df_part)
end

function read_Danon_2013_stratified_dds()
	df, df_part = read_danon_2013_contact(; fil_adult = true);
	cond = df_part[:, :P_Post] .== 1
	df_part_post = df_part[cond, :]
	df_part_online = df_part[.!cond, :]
	cond = in.(df[:, :part_id_d], Ref(df_part_post[:, :part_id_d]))
	df_post = df[cond, :]
	df_online = df[.!cond, :]

	df_tmp = danon_degree_dist_for_all_hm_nhm(df_post, df_part_post)
	df_dd_post = add_zero_counts_to_df_deg(df_tmp, df_part_post)
	df_dd_post[:, :key] .= "Danon_2013_post"

	df_tmp = danon_degree_dist_for_all_hm_nhm(df_online, df_part_online)
	df_dd_online = add_zero_counts_to_df_deg(df_tmp, df_part_online)
	df_dd_online[:, :key] .= "Danon_2013_online"
	return (vcat(df_dd_post, df_dd_online), vcat(df_part_post, df_part_online))
end

"""
Note: including group contacts, the number of those with 0 total contacts are zero.
"""
function read_danon_2013_contact(; fil_adult = false)
	df_danon = CSV.read("../dt_Leon_Danon_2013/Contact_data.csv", DataFrame)
	@rename!(df_danon, :part_id_d = :C_PID)
	df_part = read_danon_2013_part(; fil_adult = fil_adult)
	# Remove children answers by inner join.
	df = innerjoin(df_danon, df_part; on = :part_id_d) |>
		 standardise_danon_2013_to_socialmixer_data
	# Remove parts with total contacts of missing.
	cond = ismissing.(df_part[:, :P_total_contacts])
	ids = df_part[cond, :part_id_d] |> unique
	df_part = df_part[.!cond, :]
	df = @subset(df, .!in.(:part_id_d, Ref(ids)))
	return (df, df_part)
end

function read_danon_2013_part(; fil_adult = false)
	df_part = CSV.read("../dt_Leon_Danon_2013/Person_data.csv", DataFrame)
	@rename!(df_part, :part_id_d = :P_ID, :part_age = :P_age,
		:part_gender = :P_gender)
	if fil_adult == true
		df_part = @subset(df_part, :part_age .>= 18)
	end

	df_part = @select(df_part, :part_id_d, :part_age, :part_gender, :P_total_contacts, :P_Post)
	df_part[:, :hh_id] .= "NA"
	return df_part
end

function standardise_danon_2013_to_socialmixer_data(df::DataFrame)::DataFrame
	df_new = @select(df, :part_id_d, :C_Wheres_1)
	@rename!(df_new,
		:cnt_home = :C_Wheres_1,
	)
	@transform!(df_new,
		:key = "Danon_2013",
		# Note: this is a temporal filling.
		:cnt_work = "NA", # Since Work/school is mixed.
		:cnt_hh = "NA",
		:cnt_school = "NA",
		:cnt_transport = "NA",
		:cnt_leisure = "NA",
		:cnt_otherplace = "NA",
		:cont_id = "NA",
	)
	return df_new
end

function danon_degree_dist_for_all_hm_nhm(df::DataFrame, df_part::DataFrame)
	df_hm = @subset(df, :cnt_home .== 1)
	df_hm = combine(groupby(df_hm, :part_id_d), nrow => :cnt)

	# Note: since some of the total contacts are missing, those participants
	#     were removed from all and non-home.
	df_deg_hm = @pipe leftjoin(df_part, df_hm, on = :part_id_d) .|>
					  coalesce(_, 0) |>
					  @rename(_, :cnt_home = :cnt) |>
					  @transform(_, :cnt_non_home = :P_total_contacts .- :cnt_home)

	# Prepare each degree distribution
	@transform!(df_deg_hm, :strat = "home")

	df_all = @chain df_deg_hm begin
		@rename :cnt = :P_total_contacts
		@select :part_id_d :cnt
		@transform :strat = "all"
		@subset :cnt .>= 1
	end
	df_nhm = @chain df_deg_hm begin
		@rename :cnt = :cnt_non_home
		@select :part_id_d :cnt
		@transform :strat = "non-home"
		@subset :cnt .>= 1
	end
	df_hm[!, :strat] .= "home"
	return vcat(df_all, df_hm, df_nhm)
end


#########################
##### CoMix UK data #####
#########################

function read_comix_uk_dds()
	df_comix, df_part = read_comix_uk_contact_adult_2021Jul_2022Mar();
	return get_df_dd_single(df_comix, df_part, "CoMix_uk_internal")
end

function read_comix_uk_contact_adult_2021Jul_2022Mar()
	function filter_date(df)
		return @subset(df, Date(2021, 7, 1) .<= :date .< Date(2022, 4, 1))
	end
	df_comix = read_comix_uk_contact() |> filter_date
	df_comix = standardise_comix_uk_to_socialmixer_data(df_comix)
	df_part = CSV.read("../dt_comix_no_public/part_uk.csv", DataFrame) |>
			  filter_date
	# Standardise columns.
	df_part = @select(df_part,
		:part_id = :part_wave_uid,
		:part_age = :part_age_group,
		:part_gender = :part_gender_nb)
	df_part[:, :hh_id] .= "NA"
	df_part = filter_adult_cate(df_part, col = :part_age)

	# Include >=18 by innerjoin.
	df_comix = innerjoin(df_comix, df_part, on = :part_id)
	return (df_comix, df_part)
end

function read_comix_uk_contact()
	df_comix = CSV.read("../dt_comix_no_public/contacts_uk.csv", DataFrame)
	df_comix[!, :year_month] = convert_date_to_year_month.(df_comix.date)
	df_comix[!, :year_q] = convert_date_to_quarter.(df_comix.date)
	return df_comix
end

function standardise_comix_uk_to_socialmixer_data(df_comix; skip_select = false)
	if skip_select == false
		df_new = @select(df_comix, :part_wave_uid, :cnt_household, :cnt_home, :cnt_work)
	else
		df_new = @select(df_comix, Not(:part_id))
	end
	@rename!(df_new,
		:part_id = :part_wave_uid,
		:cnt_hh = :cnt_household,
		:cnt_home = :cnt_home,
		:cnt_work = :cnt_work,
	)
	@transform!(df_new,
		:key = "CoMix_uk_internal",
		# TODO: this is a temporal filling.
		:cnt_school = "NA",
		:cnt_transport = "NA",
		:cnt_leisure = "NA",
		:cnt_otherplace = "NA",
		:cont_id = "NA",
	)
	return df_new
end

#########################
##### CoMix2 data   #####
#########################

function read_comix2_df_and_df_part()
	df_master = read_survey_master_data();
	key = "CoMix2"
	r_survey = @subset(df_master, :key .== key)[1, :];

	df, df_part = read_raw_sc_data(r_survey);
	return(df, df_part)
end

function read_comix2_dds()
	df, df_part = read_comix2_df_and_df_part()
	@transform!(df_part, :country = map(x -> x[1:2], :part_id));
	df_part = filter_adult_cate(df_part; col = :part_age)
	df = innerjoin(df, df_part, on = :part_id)
	df_dd, df_part = get_df_dd_single(df, df_part, key);
	return (df_dd, df_part)
end

function read_comix2_stratified_dds()
	df, df_part = read_comix2_df_and_df_part()
	@transform!(df_part, :country = map(x -> x[1:2], :part_id));

	df_dd_mer = DataFrame()
	df_part_mer = DataFrame()
	for cnt in df_part[:, :country] |> unique
		df_part_tmp = @subset(df_part, :country .== cnt)
		df_part_tmp = filter_adult_cate(df_part_tmp; col = :part_age)
		df_tmp = innerjoin(df, df_part_tmp, on = :part_id)
		df_dd, df_part_tmp = get_df_dd_single(df_tmp, df_part_tmp, string(key, "_", cnt));
		df_dd_mer = vcat(df_dd_mer, df_dd)
		df_part_mer = vcat(df_part_mer, df_part_tmp)
	end
	return (df_dd_mer, df_part_mer)
end

################################
##### Descriptive analysis #####
################################

function stacked_bar_cate(df_dds)
	tab = @pipe groupby(df_dds, [:key, :strat]) |> combine(_, :y => sum => :n_part)
	rem_lis = ["Danon_2013", "Leung_2017",  #"CoMix2",
		"CoMix2_at", "CoMix2_be", "CoMix2_dk",
		"CoMix2_ee", "CoMix2_gr", "CoMix2_hr",
		"CoMix2_it", "CoMix2_pl", "CoMix2_pt",
	]
	tab = @subset(tab, @byrow (:key in rem_lis) == false) ;
	tab = @subset(tab, :strat .== "all");
	# meta data
	df_master = read_survey_master_data();
	add_vis_cols_to_master!(df_master)
	df_mas = @select(df_master, :key, :group_c, :mode, :cutoff_less90);

	tab = leftjoin(tab, df_mas, on = :key);
	# Arrang dataframes
	tab = @transform(tab,
		@byrow :n_part_cate = :n_part < 1000 ? "0-1000" :
			:n_part <= 5000 ? "1000-5000" :
			:n_part <= 10000 ? "5001-10000" :
			:n_part > 10000 ? "10000+" : "NaN"
	)
	tab = @transform(tab, @byrow :group_c_pri = (:group_c == "G-Yes") ? "+ group" : "")
	tab = @transform(tab,
		@byrow :mode_pri = :mode == "P" ? "Paper or Interview" :
			:mode == "O" ? "Online" :
			:mode == "PI" ? "Paper or Interview" :
			:mode == "I" ? "Paper or Interview" : "NaN"
	)
	tab = @transform(tab, :cate = string.(:mode_pri, :group_c_pri))
	sort(tab, :n_part_cate);

	tab_plot = @pipe groupby(tab, [:n_part_cate, :cate]) |> combine(_, nrow => :n_cnt)
	tab_plot = vcat(tab_plot,
		DataFrame(n_part_cate = "5001-10000", cate="Online", n_cnt=0)
	)
	n_part_level = ["0-1000", "1000-5000", "5001-10000", "10000+"]
	tab_plot[!, :n_part_cate_level] = categorical(
		tab_plot[:, :n_part_cate],levels=n_part_level, ordered=true)

	pl = @df tab_plot groupedbar(:n_part_cate_level, :n_cnt,
		group=:cate, bar_position=:stack,
		xticks=([1,2,3,4], n_part_level)
	)
	plot!(pl, xlabel="Number of answers", ylabel= "Number of survey")
end

function clean_survey_key_names!(df_dds::DataFrame)
	df_dds[!, :key] = replace(df_dds[:, :key], "CoMix2" => "CoMix2 All")
	@transform!(df_dds, :key = replace.(:key,
		"CoMix_uk_internal" => "CoMix UK",  # TODO: update CoMix_uk_internal to CoMix UK.
		"post" => "(paper)",
		"paper" => "(paper)",
		"online" => "(online)",
		"_" => " "
	));
	@transform!(df_dds, :key = transform_comix2.(:key));
	nothing
end

function transform_comix2(s::String)
    if startswith(s, "CoMix2 ") && length(s) == 9
        prefix = s[1:7]
        suffix = uppercase(s[8:9])
        return prefix * suffix
    else
        return s
    end
end