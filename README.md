# Graph-matching benchmark (Boscia / MIP)

Julia code for benchmarking **graph matching** and related solvers built around **Boscia** (branch-and-bound + Frank–Wolfe over the Birkhoff polytope). Continuous baselines that target plain graph isomorphism (spectral, penalty Frank–Wolfe, DCA) are not included here.

The main entrypoint is `test_example.jl`, which calls `bench(...)` from `benchmarkProblems.jl`.

## What’s in this folder

- **`benchmarkProblems.jl`**: `bench(graph, seed; solver=..., time_limit=..., write=...)` loads an instance, builds two random relabelings of the same graph (a feasible graph-matching instance), runs the chosen solver, and optionally writes a CSV result.
- **`bosciaGraphIsomorphism.jl`**: Boscia-based graph-matching solver, including optional preprocessing (clique, star, OBBT).
- **`mip.jl`**: a MIP formulation (SCIP / HiGHS via JuMP).
- **`utilities.jl`**: graph loading and helper utilities (`load_graph`, `randomPermutation`, …).
- **`test_example.jl`**: a minimal runnable example (edit solver/options here).

## Data / folder layout (expected by `load_graph`)

`utilities.jl`’s `load_graph(name; format=...)` expects these folders (relative to this directory):

- **MAT instances**: `./export_mat/<name>.mat`
- **DIMACS instances**: searched under `./more_benchmark/`
- **Cospectral non-iso DIMACS instances**: searched under `./cospectral_benchmark/`

If you use your own dataset layout, update `load_graph` in `utilities.jl`.

## Install & run

From this directory, start Julia and install the required packages (the code uses `Boscia`, `FrankWolfe`, `JuMP`, `SCIP`, `HiGHS`, `MAT`, `CSV`, `DataFrames`, etc.):

```julia
import Pkg
Pkg.activate(".")
Pkg.add([
  "Boscia", "Bonobo", "FrankWolfe", "CombinatorialLinearOracles",
  "JuMP", "SCIP", "HiGHS",
  "MAT", "CSV", "DataFrames",
])
```

Run the example:

```bash
julia test_example.jl
```

## Solver strings (how `bench` interprets `solver`)

`bench(...; solver="...")` dispatches based on substrings in the solver name:

- **Boscia**: any solver string containing `"boscia"` triggers `boscia_run(...)` (graph-matching formulation).
  - **Traversal**: `"DFS"` enables depth-first traversal; `"left"` can be used in the name to prefer the left branch in depth-first mode.
  - **Frank–Wolfe variant**: `"fw"` / `"bpcg"` / `"dicg"` selects the FW variant (DICG is the default in `boscia_run`).
  - **Preprocessing**: append **`rdc`** to run Frank–Wolfe reduced-cost fixing before branch-and-bound (see `FW_reduced_cost_preprocess` in `bosciaGraphIsomorphism.jl`). Example: `solver="boscia_DFS_rdc"`.
- **MIP**: any solver containing `"mip"` runs the MIP approach in `mip.jl`.

## Output (optional CSV)

If `write=true`, `bench` writes a one-row CSV under a solver-specific output folder (see the `result_path` in `benchmarkProblems.jl`). If you want portable output, change `result_path` to a local directory.
