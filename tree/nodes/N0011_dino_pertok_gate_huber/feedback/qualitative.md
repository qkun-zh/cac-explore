# Qualitative Feedback — N0011_dino_pertok_gate_huber

## Naming Conventions Audit
- **Slug** `N0011_dino_pertok_gate_huber` follows `N####_<slug>`; descriptive (per-token gate + Huber) though `pertok` typo vs `pertoken` is minor.
- **Class** `PerTokenGateMLP` (`model.py:30`) — descriptive CamelCase, matches idea.md spec; `in_dim=768, hidden=64` clear. Reused `DinoPromptV2` / `PromptEncoderV2` collides with N0007/N0010 but isolation via engine import avoids runtime clash.
- **Constants** `PATCH=14` (`model.py:9`), `BACKBONE` (`model.py:8`) UPPER_SNAKE at module top, consistent with N0010.
- **Config keys** `gate_mlp_hidden`, `loss_function="huber"`, `huber_delta` well-named; BCHW handling (`model.py:79-81`) correctly branches `ndim==3` vs 4.

## Docstring Presence & Self-Containedness
- Module docstring line 1 present, self-contained summary.
- `PerTokenGateMLP` has one-line docstring (`model.py:31`) — improvement over N0010's missing docs.
- `PromptEncoderV2`, `DinoPromptV2`, `build_model` lack class/method docstrings; `forward()` gate logic (`model.py:84-85`) and BCHW reshape lack inline rationale. Readable but under-documented for a two-change node.

## MODIFICATIONS.md / Notes Clarity
- No `MODIFICATIONS.md` exists (verified). Per `PROTOCOL.md:§2` not required; idea.md `## Delta vs Parent` covers scalar→MLP gate + MSE→Huber (`code/engine/train.py:149` via `F.huber_loss`). Suffices for isolation, but separate changelog would aid synthesis disambiguation (two variables confounded).

## Hypothesis Format Compliance
- **H0019/H0020** both in `IF [choice] IN [scope], THEN [effect], BECAUSE [mechanism]. DISPROVED IF [criterion].` per `AGENTS.md:107` — compliant, scoped to multi-tap DINOv2, mechanistic, falsifiable (MAE≥ parent; RMSE/MAE >3.5). Gap in H0020 (3.0 pass vs 3.5 fail) leaves neutral zone intentionally.

## Idea Clarity & Falsifiability
- Motivation clear: targets N0010's 3.6× RMSE/MAE outlier tail with two surgical changes (per-token spatial gating + Huber delta=5 at engine line 149). Spec precise (`core_ideas`/`invariants`/`tunable_aspects`), params 23.16M <32M. Falsification decisive: best MAE 26.68 >21.53 (regression +5.15), RMSE/MAE 3.51 → both hypotheses disproved (H0019 contradiction, H0020 neutral/fail).

## Overall Assessment
| Dimension | Verdict | Notes |
|-----------|---------|-------|
| Naming | ✅ | PerTokenGateMLP clear; PATCH/BCHW good; legacy class reuse |
| Docstrings | ⚠️ | Module+gate documented; rest missing |
| MODIFICATIONS | — | Absent; Delta covers but confound remains |
| Hypothesis fmt | ✅ | H0019/H0020 compliant & falsifiable |
| Idea clarity | ✅ | Surgical, well-scoped; disproved cleanly |
