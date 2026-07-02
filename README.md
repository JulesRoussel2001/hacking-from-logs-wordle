# Detecting Reward Hacking from Behavior Logs (Off-Policy, Wordle/RLVR)

**Question.** In agentic RL, can *off-policy* diagnostics — computed only from an
old behavior policy's logs — flag that a new policy is **reward hacking**, as
opposed to merely being *different* (benign drift)? Most OPE diagnostics were
built to answer "are these logs reliable for evaluating this policy?"; whether
any of them can answer "is this policy hacking?" is the open question here.

This repo is a controlled testbed for that question on real
[TextArena](https://github.com/LeonGuertler/TextArena) Wordle, with GRPO-trained
LLM policies (Qwen2.5-0.5B-Instruct) and gated, audited measurement throughout.
It is the off-policy sequel to my measurement-layer study of reward/metric/
extractor fusion in open RLVR
([post](https://www.lesswrong.com/posts/hBjn9rqgjrktH9LL3/in-open-rlvr-improvement-depends-on-the-instrument-a-small-2),
[repo](https://github.com/JulesRoussel2001/grpo-reward-vs-eval)).

## Experiment design

| Policy | Training reward | Gate before admission |
|---|---|---|
| **A** (behavior/logging) | solve (env-verifiable truth), ε-wrapped for logging | — |
| **B_hack** | Gate-1-admissible **pure proxy** (`tiles`), optimizing **mean_proxy_return** (per-turn mean, never the sum) | **Gate 2**: proxy ↑ vs A **and** solve ↓ vs A, on-policy |
| **B_drift** | solve, with one benign knob varied (seed / temperature / KL / legitimate shaping) | **drift gate**: solve within tolerance of A |

Logs come from A. OPE (per-decision IS, per-decision SNIS, trajectory SNIS,
with per-turn ESS and coverage probes) evaluates B_hack and B_drift from those
logs; on-policy evaluation is the controlled ground truth. The final study
divergence-matches drift to hacked (with a match-quality gate) and classifies
each diagnostic as: candidate hacking-specific signal / distance tracker /
anti-signal / no reliable signal. **Both outcomes are informative** — a clean
negative ("OPE diagnostics detect distribution shift, not hacking") is a
result, not a failure.

## Design decisions worth knowing

- **Pure proxies, gated twice.** A flawed reward must look reasonable on
  faithful play (Gate 1: lift over random + correlation with graded truth on
  the behavior policy's episodes) and only diverge under optimization (Gate 2).
  No ground truth is ever blended into the training proxy. `yellow_only` is
  kept as a **negative control** the gate rejects; it can never be selected as
  a training reward.
- **Constrained action space.** Policies are distributions over the env's own
  secret vocabulary (156 words, derived from TextArena's code path), minus
  already-guessed words (the env rejects repeats). LLM action probabilities are
  softmax over per-word sequence logprobs — raw free-form text probability is
  structurally unable to reach importance sampling. This is a **named
  restriction**: the real env accepts a larger guess dictionary.
- **Valid IS denominators.** The logging policy is an explicit ε-mixture; logs
  store both `model_p` and `behaviour_p` (the final sampling probability, the
  only valid denominator), plus schema/eps/N/prompt-hash metadata.
- **Environment quirks are handled, tested, and documented**: exact
  duplicate-aware feedback (cross-checked 200×), the env prints *no feedback
  row on a win*, repeated guesses are rejected, invalid guesses can never
  become fake feedback, secrets are seedable for byte-identical reruns.
- **Custom GRPO loop** (group-relative advantages, KL to frozen reference)
  rather than TRL's `GRPOTrainer`, because constrained action sampling — not
  free-form generation — is required for clean OPE.

## Quickstart (CPU, ~5 min)

```bash
pip install -r requirements.txt
python wordle_gpu_phase.py selftest   # exact-feedback, guards, estimators, ...
python wordle_gpu_phase.py mock       # FULL pipeline with mock policies
```

The mock run executes every stage — Gate 1, on-policy tables, Gates 2/drift,
logging, OPE tables, the hacked-vs-drift study — with fake policies. **Every
number it prints is a code-path check, not a result** (the mock Gate 2
deliberately demonstrates the abort path). Reference output:
`docs/mock_pipeline_run.log`.

## GPU phase (Colab A100, ~1 day)

```bash
python wordle_gpu_phase.py runbook
```

prints the ten steps (train A → Gate 1 → train B_hack → train B_drift →
verify gates → log → OPE → study), with hard stops at every gate. Cell-by-cell
Colab notes: `docs/colab_runbook.md`.

## Status & claim ladder

**Status: CPU-verified harness; GPU results pending.** Claims are kept on a
ladder and never conflated: (1) harness validation ✅ (self-tests, byte-identical
reruns) → (2) proxy admissibility under trained A → (3) learned hacking
emergence (Gate 2) → (4) OPE estimation accuracy vs on-policy truth →
(5) hacking-vs-drift diagnostic specificity. "Hacking detected" may only ever
be claimed for a diagnostic that separates B_hack from *divergence-matched*
benign drift **and** tracks the proxy–truth gap rather than policy distance.

**Caveats** (from the study output, kept verbatim): single seed,
ANSWERS-restricted action space, one proxy design, TextArena vocabulary,
small-scale GRPO. `textarena` is pinned to 0.7.4 because the harness encodes
version-specific env behavior — rerun `selftest` before any upgrade.

## Repo map

- `wordle_real_stack.py` — CPU harness: env adapter, exact feedback,
  policies, Gate 1, OPE estimators, self-tests.
- `wordle_gpu_phase.py` — experiment layer: HF policy, GRPO loop, gates,
  logging, OPE tables, hacked-vs-drift study, mock pipeline, CLI.
- `docs/` — Colab runbook, reference mock output.
- `results/` — (created when they exist) gate reports and study tables from
  real runs, committed with their configs.

MIT license.
