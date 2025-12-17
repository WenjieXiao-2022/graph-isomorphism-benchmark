# GraphIsomorphismsWithBoscia

Julia code for benchmarking **graph isomorphism (GI)** solvers, with a focus on **Boscia** (branch-and-bound + Frank–Wolfe over the Birkhoff polytope).

The main entrypoint is `test_example.jl`, which calls `bench(...)` from `benchmarkProblems.jl`.

## What’s in this folder

- **`benchmarkProblems.jl`**: `bench(graph, seed; solver=..., format=..., time_limit=..., iso_generate=...)` loads an instance, generates an isomorphic or non-isomorphic pair, runs the chosen solver, and optionally writes a CSV result.
- **`bosciaGraphIsomorphism.jl`**: Boscia-based GI solvers, including optional preprocessing (clqiue, star and OBBT) routines.
- **`spectral.jl`**: spectral + assignment-based isomorphism check (`isIsomorphic`, `isIsomorphicRepeated`) using the Hungarian algorithm.
- **`mip.jl`**: a MIP formulation of GI (SCIP / HiGHS via JuMP).
- **`penalty.jl`**, **`dca.jl`**: additional continuous optimization baselines.
- **`utilities.jl`**: graph loading and helper utilities (`load_graph`, `randomPermutation`, `non_iso_graph`, …).
- **`test_example.jl`**: a minimal runnable example (edit solver/options here).

## Data / folder layout (expected by `load_graph`)

`utilities.jl`’s `load_graph(name; format=...)` expects these folders (relative to this directory):

- **MAT instances**: `./export_mat/<name>.mat`
- **DIMACS instances**: searched under `./more_benchmark/`
- **Cospectral non-iso DIMACS instances**: searched under `./cospectral_benchmark/`

If you use your own dataset layout, update `load_graph` in `utilities.jl`.

## Install & run

From this directory, start Julia and install the required packages (the code uses `Boscia`, `FrankWolfe`, `Graphs`, `NautyGraphs`, `Hungarian`, `JuMP`, `SCIP`, `HiGHS`, `MAT`, `CSV`, `DataFrames`, etc.):

```julia
import Pkg
Pkg.activate(".")
Pkg.add([
  "Boscia", "Bonobo", "FrankWolfe", "CombinatorialLinearOracles",
  "Graphs", "NautyGraphs", "Hungarian",
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

- **Boscia**: any solver string containing `"boscia"` triggers `boscia_run(...)`.
  - **Traversal**: `"DFS"` enables depth-first traversal; `"left"` can be used in the name to prefer the left branch in depth-first mode.
  - **Frank–Wolfe variant**: `"fw"` / `"bpcg"` / `"dicg"` selects the FW variant (DICG is the default in `boscia_run`).
  - **Preprocessing (warm-start)**: append any of:
    - `"clique"`: clique-based fixings
    - `"star"`: star-based fixings
    - `"OBBT"`: bound-tightening based fixings
    Example: `solver="boscia_DFS_clique_star_OBBT"`.
- **Nauty**: `solver="nauty"` uses `NautyGraphs` to test isomorphism.
- **Spectral**: `solver="spectral"` uses the spectral + Hungarian assignment approach.
- **MIP**: any solver containing `"mip"` runs the MIP approach in `mip.jl`.
- **DCA**: any solver containing `"dca"` runs the DCA baseline (`dca.jl`).
- **Penalty/FW**: any solver containing `"penalty"` runs the penalty/FW baseline (`penalty.jl`).

## Generating non-isomorphic pairs

`bench(...; iso_generate=false)` creates a non-isomorphic instance by **flipping edges** (see `non_iso_graph` in `utilities.jl`).
You can encode the number of flips as a trailing suffix in the solver string, e.g. `solver="boscia_DFS_5"` to flip 5 edges.

## Output (optional CSV)

If `write=true`, `bench` writes a one-row CSV under a solver-specific output folder (see the `result_path` in `benchmarkProblems.jl`). If you want portable output, change `result_path` to a local directory.
