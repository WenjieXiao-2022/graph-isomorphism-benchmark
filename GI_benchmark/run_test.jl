using LinearAlgebra

include("./benchmarkProblems.jl")

# ------------------------------------------------------------
# Solver options
#
# Boscia-based solvers (single string in `solver` argument):
#   "boscia_fw"
#   "boscia_bpcg"
#   "boscia_dicg"
#   "boscia_dicg_depth"
#   "boscia_dicg_depth_left"
#   "boscia_dicg_frac32"
#   "boscia_dicg_depth_frac32"
#
# Keywords:
#   - "depth"     : use depth-first strategy in the BnB tree
#   - "frac32"    : use ||XA-BX||_F^(3/2) objective variant
#   - "left"      : modify branching/traversal to favor left children
#
# Graph-matching variants:
#   add "gm" to the solver name, optionally with a flip count suffix:
#       "boscia_dicg_gm_5"
#       "boscia_dicg_depth_gm_10"
#   The trailing number indicates how many edges to flip; if that
#   number exceeds the number of edges, all edges are flipped.
#
# Non-isomorphism (non-iso) variants:
#   add "noniso" similarly:
#       "boscia_dicg_noniso_5"
#       "boscia_dicg_depth_noniso_10"
#
# MIP solvers:
#   "mip"
#   "mip_l1"
#   "mip_l1_nosym"
#
# Keywords:
#   - "nosym" : disable symmetry computation
#   - "gm"    : graph-matching instances
#   - "noniso": non-isomorphism instances
#
# Nauty:
#   To run nauty, use:
#       solver = "nauty"
#
# Symmetry generation:
#   For non-iso and gm cases, set:
#       iso_generate = false
#   For standard GI instances, you can keep:
#       iso_generate = true
# ------------------------------------------------------------

# Example: run Boscia DICG on a single instance
bench(
    "exact_001",  # instance name / ID in your logging
    3,            # number of repetitions
    ".",
    format      = "dimacs",
    verbose     = true,
    solver      = "boscia_dicg",
    write       = false,
    iso_generate = true,
    time_limit  = Inf,
)