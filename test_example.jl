include("benchmarkProblems.jl")


# ------------------------------------------------------------
# Solver options
#
# Boscia-based solvers:
#   - "boscia"
#   - "boscia_DFS"
#   - "boscia_DFS_left"
#
#   Preprocessing:
#     Append "rdc" to run Frank–Wolfe reduced-cost fixing before branch-and-bound.
#     Example: "boscia_DFS_rdc"
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
# Other available solvers:
#   - "mip"
#
# ------------------------------------------------------------
# GOE synthetic instances (is_GOE_graph = true)
#
#   These use the Gaussian Orthogonal Ensemble (GOE): a random real symmetric
#   matrix A with independent upper-triangular entries, diagonal N(0, 2/n) and
#   off-diagonal N(0, 1/n) (see generate_wigner_goe in utilities.jl).
#
#   generate_easy_boscia_instance builds a planted graph-matching pair:
#     - A_noisy = A + sigma * Z, with Z another independent GOE draw
#     - sigma = n^(noise_threshold - epsilon)  (defaults: noise_threshold=-1,
#       epsilon=0.1 in utilities.jl; bench passes noise_threshold_GOE)
#     - B is the same matrix A_noisy after the same row and column permutation
#       (planted map P in Pi_star / true_mapping).
#
#   bench still passes the clean A as the first matrix and B as the second
#   (see benchmarkProblems.jl). The `graph` argument is ignored for the instance
#   but can still label CSV output when write = true.
#
#   Solver names may include the substring "GOE" for your own bookkeeping; only
#   tokens like DFS, left, rdc, fw, bpcg change Boscia’s behavior.
#
# ------------------------------------------------------------

# Example: run Boscia DICG on a single instance

bench(
    "latin_3_9",  # graph name
    3;        # random seed
    solver      = "boscia_GOE",
    time_limit  = 3600,
    write       = false,
    is_GOE_graph = true,
    num_nodes_GOE = 50,
    # noise_threshold_GOE = -0.5,
)
