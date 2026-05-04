using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using GI_benchmark


# ------------------------------------------------------------
# Solver options
#
# Boscia-based solvers:
#   - "boscia"
#   - "boscia_DFS"
#   - "boscia_DFS_left"
#
#   Preprocessing methods:
#     The following preprocessing techniques are supported:
#       • clique
#       • star
#       • OBBT
#
#     Preprocessing can be activated by appending the corresponding
#     keywords to the solver name. For example:
#       - "boscia_DFS_star"
#       - "boscia_DFS_clique_star"
#
#   Frank–Wolfe (FW) variants:
#     Different FW variants can be selected. By default, DICG is used.
#     The classical FW and BPCG variants are also supported.
#
#     The FW variant can be selected by appending the corresponding
#     keyword to the solver name. For example:
#       - "boscia_fw"
#       - "boscia_bpcg"
#
# Non-isomorphism testing:
#   Set `iso_generate = false` and specify the number of edges to flip
#   directly in the solver name. For example:
#       - "boscia_DFS_5"
#
# Graph-matching (GM) variants:
#   - Add the suffix "GM" to any Boscia-based solver name, optionally
#     followed by an integer specifying the number of edge flips.
#
#   Examples:
#       - "boscia_DFS_GM_5"
#       - "boscia_DFS_left_GM_10"
#
#   The trailing integer indicates the number of edges to flip. If this
#   number exceeds the total number of edges in the graph, all edges
#   are flipped.
#
# Other available solvers:
#   - "nauty"
#   - "mip"
#   - "dca"
#   - "penalty"
#
# ------------------------------------------------------------

# Example: run Boscia DICG on a single instance

GI_benchmark.bench(
    "cfi-20",  # graph name
    3;        # random seed
    solver      = "boscia_DFS_expwalk",
    time_limit  = 3600,
    iso_generate = true,
)
