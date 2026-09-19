"""Composition-feasibility registry (mechanism switch conflict detector).

Some hypotheses prescribe mutually exclusive behaviour on the SAME head
component or SAME config switch — e.g. H0009 (content-conditioned 33,024-param
exemplar channel gate) and H0011 (the identical gate fed a frozen random
vector) configure the same gate component; both H0003 and H0007 select the
backbone readout regime. Booking such a pair into one child node yields a
structurally infeasible or un-interpretable joint test.

This module is the LOCAL addition (recorded deviation, see AGENTS §9) that
guards node COMPOSITION only. It never changes the scoring of Eq.5-6 —
candidates are still ranked by dual Thompson-exploit + epistemic-explore — it
only refuses to co-compose hypotheses that physically cannot coexist.

Conflict rule: two hypotheses conflict iff they share at least one component
key or at least one config-switch key. Unknown ids default to no components /
no switches and therefore never conflict (conservative: only drop KNOWN
mutually-exclusive pairs).

Feasibility also requires PARENT compatibility: a hypothesis may declare
``requires`` — config switches the reasoning unit assumes are already present
in the parent (e.g. H0011 severs the input of an EXISTING channel gate, so its
child must inherit use_channel_gate=true). Adjoins whose requirements the
parent config does not satisfy are dropped at booking time.
"""
from __future__ import annotations


def _reg() -> dict[str, dict]:
    # id -> {components: set[str], switches: set[str], requires: set[str]}
    return {
        "H0001": {"components": {"aux_count_head"}, "switches": {"use_global_count_aux"}, "requires": set()},
        "H0002": {"components": {"exemplar_summary"}, "switches": {"use_coarse_exemplar_summary"}, "requires": set()},
        "H0003": {"components": {"backbone_readout"}, "switches": {"use_stage23_readout"}, "requires": set()},
        "H0004": {"components": {"condenser"}, "switches": {"use_condenser"}, "requires": set()},
        "H0005": {"components": {"backbone_regime"}, "switches": {"unfreeze_backbone"}, "requires": set()},
        "H0006": {"components": {"fine_fuser"}, "switches": {"use_dilated_branch"}, "requires": set()},
        "H0007": {"components": {"backbone_readout"}, "switches": {"use_final_readout"}, "requires": set()},
        "H0008": {"components": {"decoder_context"}, "switches": {"use_spatial_summaries"}, "requires": set()},
        "H0009": {"components": {"channel_gate"}, "switches": {"use_channel_gate"}, "requires": set()},
        "H0010": {"components": {"exemplar_gate"}, "switches": {"use_exemplar_gate"}, "requires": set()},
        "H0011": {"components": {"channel_gate"}, "switches": {"use_const_gate_input"}, "requires": {"use_channel_gate"}},
        "H0012": {"components": {"condenser_temperature"}, "switches": {"use_count_temp"}, "requires": set()},
        "H0013": {"components": {"upsample"}, "switches": {"use_subpixel_up"}, "requires": set()},
    }


REGISTRY: dict[str, dict] = _reg()


def components(hyp_id: str) -> frozenset:
    return frozenset(REGISTRY.get(hyp_id, {}).get("components", []))


def switches(hyp_id: str) -> frozenset:
    return frozenset(REGISTRY.get(hyp_id, {}).get("switches", []))


def requires(hyp_id: str) -> frozenset:
    return frozenset(REGISTRY.get(hyp_id, {}).get("requires", []))


def conflicts(a: str, b: str) -> bool:
    if a == b:
        return False
    ca, cb = components(a), components(b)
    if ca & cb:
        return True
    sa, sb = switches(a), switches(b)
    if sa & sb:
        return True
    return False


def feasible(ids: list[str]) -> list[str]:
    """Greedily keep ids (in order), dropping any that conflicts with a kept one."""
    kept: list[str] = []
    for h in ids:
        if any(conflicts(h, k) for k in kept):
            continue
        kept.append(h)
    return kept