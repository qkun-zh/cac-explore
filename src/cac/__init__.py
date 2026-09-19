"""cac — Active Hypothesis Exploration for Crowd Counting.

A restructured lineage of the core `hypoexplore` machinery: the trajectory
tree IS the directory tree under `tree/`; the hypothesis memory is an
append-only ledger; the paper's selection/confidence math lives in
`cac.expt` and is reachable only through the discovery CLI. See
AGENTS.md / README.md at the repo root.
"""
__version__ = "0.1.0"