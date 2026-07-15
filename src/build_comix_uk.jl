# build_comix_uk.jl — one-time CoMix → CoMix-UK data build.
#
# Mirrors the notebook `1j_data_explore.ipynb` cells but with a lean preamble
# (no Turing/Plots) so it runs headless in seconds instead of minutes.
#
#   raw  dt_comix_no_public/{contacts,part}.csv   (all 20 countries)
#     -> filter to country == "uk"
#     -> dt_comix_no_public/{contacts_uk,part_uk}.csv
#     -> dt_comix_no_public/{contacts_uk,part_uk}.arrow   (mmap'd by readers)
#
# Run from src/:  julia --project=/workdir src/build_comix_uk.jl

using CSV
using Arrow
using DataFrames
using DataFramesMeta
using XLSX

const DIR = "../dt_comix_no_public"

# ---------------------------------------------------------------------------
# 1. Variable tables (data dictionary). The `final_part` / `final_contact`
#    sheets document every column; `country` (final_part row 1) is the
#    "Country 2-letter abbreviation" we filter on.
# ---------------------------------------------------------------------------
dd_path = joinpath(DIR, "dd_v1238_20230328.xlsx")
dd_contact = DataFrame(XLSX.readtable(dd_path, "final_contact"))
dd_part    = DataFrame(XLSX.readtable(dd_path, "final_part"))
@info "Variable tables" contact_vars = nrow(dd_contact) part_vars = nrow(dd_part)

# ---------------------------------------------------------------------------
# 2. Filter each raw table to the UK panel and write the `_uk.csv` files.
# ---------------------------------------------------------------------------
function filter_uk_to_csv(name::AbstractString)
    src = joinpath(DIR, name * ".csv")
    dst = joinpath(DIR, name * "_uk.csv")
    @info "Reading raw CSV" src
    df = CSV.read(src, DataFrame)
    df_uk = @subset(df, :country .== "uk")
    @info "Filtered to UK" name total = nrow(df) uk = nrow(df_uk)
    CSV.write(dst, df_uk)
    @info "  ✓ wrote CSV" dst
    return dst
end

for name in ("contacts", "part")
    filter_uk_to_csv(name)
end

# ---------------------------------------------------------------------------
# 3. Convert the UK CSVs → Arrow (mmap'd by `read_arrow_df`). Idempotent.
# ---------------------------------------------------------------------------
for name in ("contacts_uk", "part_uk")
    csv   = joinpath(DIR, name * ".csv")
    arrow = joinpath(DIR, name * ".arrow")
    @info "Converting CSV → Arrow" csv arrow
    Arrow.write(arrow, CSV.read(csv, DataFrame))
    @info "  ✓ wrote Arrow" arrow
end

@info "Done."
