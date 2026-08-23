# cac-explore — Hypothesis-Driven Multi-Agent Discovery for CAC

## Mission
> **≤32M total params achieving same-parameter-class SOTA MAE on FSC147 test.** Full fine-tuning allowed.

## Current Best
> **val MAE 20.44 / RMSE 83.06** — N0018_dino_partialft @ 23.26M params, 24min training
>
> Frozen DINOv2-S reg4 + partial FT (blocks 10-11) + multi-layer taps + area-prompt

## The Method
Implements [HypoExplore](https://arxiv.org/abs/2604.12999): hypothesis-driven scientific inquiry.
Every experiment tests 1-2 falsifiable hypotheses. Confidence scores accumulate across experiments.
Confirmed hypotheses become building blocks; refuted ones are never retried.

## Quick Start
```bash
git pull --ff-only && cat AGENTS.md STATE.md
```
