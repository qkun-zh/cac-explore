"""Single definition of the discovery-machinery constants (paper defaults).

These are the numbers that make the evolutionary machinery work. They are
paper-faithful and are NOT to be silently re-tuned by any agent. If a value
ever needs to change it changes HERE, in AGENTS.md's "constants" table, and in
one journal entry — nothing else.
"""
from __future__ import annotations

# --- bootstrap ---------------------------------------------------------------
K_BOOTSTRAP = 5          # #root proposals per bootstrap call (seed root uses migration instead)
# --- parent selection (HypoExplore Eq. 2-4) -----------------------------------
LAMBDA_ACC = 0.85        # weight on normalized accuracy in `quality`
LAMBDA_PARENT = 0.60     # weight on quality (vs availability) in `score`
TAU_MAX_SECONDS = 1800   # 30 min — max allowed training time for one node (FSC147 32ep ≈ 29 min)
# --- dual hypothesis selection (Eq. 5-6) ---------------------------------------
K_HYPO = 2               # top-k per exploit / explore subset  -> |Q_t| <= 2*K_HYPO
BETA_ALPHA0 = 1.0        # Beta prior pseudo-counts (uniform)
BETA_BETA0 = 1.0
# --- confidence update (Eq. 1) -------------------------------------------------
ETA = 0.20               # learning rate on confidence
CONF_INIT = 0.50         # maximum uncertainty
CONF_FLOOR = 0.01
CONF_CEIL = 0.99
CONF_CONFIRMED = 0.75
CONF_REFUTED = 0.25
# --- synthesis / feedback -------------------------------------------------------
K_SYNTH = 2              # max NEW hypotheses per node (updates to existing preferred)
EVIDENCE_TYPES = ("supports", "contradicts", "neutral")
# --- redundancy filter ------------------------------------------------------------
REDUNDANCY_K = 3        # top-k retrieved candidates for novelty judgement
REDUNDANCY_RETRIES = 2  # regeneration retries after a duplicate
# --- executor ----------------------------------------------------------------------
SANITY_EPOCHS = 5       # sanity-check epochs before the real run
CODE_FIX_RETRIES = 3    # budget-aware coding fix rounds (paper R_max=10, see AGENTS §9)