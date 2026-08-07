# run_nb_headless.jl — execute a notebook's code cells IN THIS PROCESS, with live stdout.
#
# WHY. `jupyter nbconvert --to notebook --execute` routes the kernel's stdout/stderr into the
# notebook's cell outputs and writes them only when the run finishes. For a 63-origin Stage-1 NUTS
# grid — days — that means `prefit_stage1!`'s per-origin heartbeat, and more importantly any
# `@warn "stage1 fit failed"`, are invisible until the very end. This runs the SAME code with the
# output going straight to a tailable log.
#
# It is not a re-implementation of the notebook: the cells are read from the .ipynb and evaluated
# one at a time in `Main`, which is exactly what a Jupyter kernel does. So it cannot drift from the
# notebook, and there is nothing to keep in sync.
#
#   julia --project=/workdir /workdir/tmp/run_nb_headless.jl 8j_preliminary_forecast.ipynb
#
# Every env knob the notebook reads (STAGE1_USE_NUTS, FIT_END, ORIGIN_MIN, S1/S2_CONCURRENCY,
# ORIGIN_12J/13J) works unchanged. cwd is set to src/ because every relative path assumes it.
#
# ⚠ This does NOT replace running the notebook under nbconvert. Plots are never rendered and no
# .ipynb output is produced, so it verifies the CODE, not the notebook. Run nbconvert afterwards —
# with the artefacts cached it is fast — to check the notebook itself still executes clean.

using Printf, Dates

const NB = length(ARGS) >= 1 ? ARGS[1] : error("usage: run_nb_headless.jl <notebook.ipynb>")
const PATH = isabspath(NB) ? NB : joinpath("/workdir/src", NB)
isfile(PATH) || error("no such notebook: $(PATH)")
cd("/workdir/src")

# Cell extraction goes through python3 rather than a Julia JSON package: JSON is not a direct
# dependency of this project (only a transitive one via IJulia), and adding one to Project.toml for
# a scratch runner would perturb the Manifest that the whole fitted grid is pinned against.
const SEP = "#=<<<CELL-BOUNDARY>>>=#"
srcs = split(read(`python3 -c """
import json,sys
cells=[c for c in json.load(open(sys.argv[1]))['cells'] if c['cell_type']=='code']
print(('\n'+sys.argv[2]+'\n').join(''.join(c['source']) for c in cells))
""" $(PATH) $(SEP)`, String), "\n$(SEP)\n")
keep = findall(s -> !isempty(strip(s)), srcs)

hms(s) = @sprintf("%dh%02dm%02ds", s ÷ 3600, (s % 3600) ÷ 60, s % 60)
say(s) = (println(s); flush(stdout); flush(stderr))

# NB_LIST_ONLY=1 prints the cell inventory and exits — a self-check that the extraction above
# parsed the notebook, without paying the multi-minute `forecast_utils.jl` load to find out.
if get(ENV, "NB_LIST_ONLY", "") == "1"
    println("$(basename(PATH)): $(length(srcs)) code cells, $(length(keep)) non-empty")
    for (n, i) in enumerate(keep)
        println(@sprintf("  cell %2d  (idx %2d, %4d chars)  %s", n, i, length(srcs[i]),
                         first(split(strip(srcs[i]), "\n"))))
    end
    exit(0)
end

say("="^96)
say("run_nb_headless | $(basename(PATH)) | $(length(keep)) non-empty code cells | pid $(getpid())")
say("  threads=$(Threads.nthreads())  started $(Dates.now())")
for k in ("STAGE1_USE_NUTS", "FIT_END", "ORIGIN_MIN", "S1_CONCURRENCY", "S2_CONCURRENCY",
          "ORIGIN_12J", "ORIGIN_13J")
    haskey(ENV, k) && say("  ENV $(k) = $(ENV[k])")
end
say("="^96)

t_all = time()
for (n, i) in enumerate(keep)
    say("\n" * "-"^96)
    say("[cell $(n)/$(length(keep))]  $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    say("-"^96)
    t0 = time()
    try
        # `include_string` into Main evaluates at top level, exactly as a kernel evaluates a cell —
        # so `let` blocks, function definitions and soft scope all behave the same.
        Base.include_string(Main, srcs[i], "$(basename(PATH)):cell$(i)")
    catch err
        say("\n*** CELL $(n) FAILED after $(hms(round(Int, time() - t0))) ***")
        showerror(stdout, err, catch_backtrace()); println(); flush(stdout)
        say("TOTAL $(hms(round(Int, time() - t_all))) — ABORTED")
        exit(1)
    end
    say("[cell $(n) done in $(hms(round(Int, time() - t0)))]")
end
say("\n" * "="^96)
say("NOTEBOOK COMPLETE: $(basename(PATH)) in $(hms(round(Int, time() - t_all)))")
say("="^96)
