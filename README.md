# Graph Isomorphism Benchmark

Julia package for benchmarking graph isomorphism (GI) solvers, with a focus on Boscia (branch-and-bound + Frank-Wolfe over the Birkhoff polytope).

The package entrypoint is `src/GI_benchmark.jl`, which exports `bench`, `load_graph`, `randomPermutation`, and `non_iso_graph`.

## Layout

- `src/GI_benchmark.jl`: module entrypoint.
- `src/benchmarkProblems.jl`: `bench(...)` orchestration and solver dispatch.
- `src/boscia.jl`: Boscia run pipeline.
- `src/boscia_preprocessing.jl`: Boscia preprocessing routines (clique/star/OBBT/walk-signature).
- `src/boscia_func_grad.jl`: Boscia objective/gradient builder functions.
- `src/spectral.jl`: spectral + Hungarian baseline.
- `src/mip.jl`: MIP baseline.
- `src/dca.jl`: DCA baseline.
- `src/penalty.jl`: penalty/FW baseline.
- `src/utilities.jl`: graph loading + utility helpers.
- `examples/example.jl`: runnable example script.
- `test/runtests.jl`: package smoke tests.

## Setup

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Run

Example:

```bash
julia examples/example.jl
```

Tests:

```julia
import Pkg
Pkg.activate(".")
Pkg.test()
```

## Solver Names

`bench(...; solver="...")` dispatches by string pattern:

- `"boscia"`: Boscia-based solver (`boscia_run`).
  - Add `"DFS"` (and optionally `"left"`) for DFS traversal behavior.
  - Add `"fw"` / `"bpcg"` / `"dicg"` for Frank-Wolfe variants.
  - Add `"clique"`, `"star"`, `"OBBT"`, `"walk"` for preprocessing.
- `"nauty"`: NautyGraphs baseline.
- `"spectral"`: spectral + Hungarian baseline.
- `"mip"`: MIP baseline.
- `"dca"`: DCA baseline.
- `"penalty"`: penalty/FW baseline.

## Data Layout

`load_graph(name)` in `src/utilities.jl` searches under:

- `./test_instances` recursively for `<name>.mat` or `<name>.dimacs`.

If your dataset structure is different, edit `load_graph` accordingly.

## Notes

- For non-isomorphic generation, use `iso_generate=false`.
- Edge flip count can be encoded as a trailing integer in solver name, e.g. `boscia_DFS_5`.
