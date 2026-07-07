# Master Table — Every Important Number of the Project
*(All values from saved logs / gate outputs / paired eval tables. Seed 41, n=200 for all paired evals unless stated. Grouped by phase.)*

## 1. Setup constants

| Quantity | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Action space | 156 words (closed, env-derived) |
| Max action entropy | ln 156 = 5.05 |
| Context per state | ~110 prompt tokens |
| Vocab (full softmax bug) | ~152k tokens |
| Compute tax of exact OPE probabilities | ~150× vs standard GRPO |
| GPU | Colab A100 40 GB (39.49 usable) |
| Colab burn rate | ~5.3 units/hour |
| Code fingerprints (last sync) | gpu_phase `ea8ec25e64af5c61` / real_stack `72962e4b2ab57f57` |

## 2. Memory (OOM #1–#4)

| Event | Key numbers |
|---|---|
| OOM #1 (full-vocab softmax + accumulated graph) | died 18→40 GB; 4.3 GB/chunk fp32; ~32 graphs alive; windowed softmax ~25× smaller |
| OOM #2 (full-fp32 overcorrection) | ~38 GB; fix = fp32 masters + bf16 autocast |
| OOM #3 (workload-dependent peak) | smoke 35.1 GiB, real run died at group ~25; 6-turn prompts ~45% longer; peak crept past 39 GiB |
| OOM #4 (chunking fallacy) | chunks 16→8 OOM'd immediately; correct fix = gradient checkpointing: **35.1 → 11.4 GiB** for ~35% more compute |
| Smoke, checkpointing + prefix cache | **12.0 GiB** peak |
| Early smoke (pre-crisis) | 22.4 GiB; scoring 0.26 s/state |

## 3. Training collapses (from scratch)

| Collapse | Numbers |
|---|---|
| #1 sparse reward | heartbeats 0.125, 0.375, then 96 consecutive zero-solve episodes (p ≈ 0.9⁹⁶ ≈ 4×10⁻⁵); ~40% of groups all-fail at ~10% solve → zero gradient |
| #2 dense reward (bf16 red herring) | objective 0.258 → 0.089 → floor; lr 2e-6 step ~40× below bf16 ulp (~8×10⁻⁵) |
| #3 probe verdict (entropy inflation) | objective 0.154 → 0.075 (floor) while entropy **2.69 → 4.42** (max 5.05); probe = 60 groups, verdict at 40 |

## 4. SFT warm start

| Run | Numbers |
|---|---|
| Teacher (consistency heuristic) | 0.98 solve |
| SFT #1 | 3k examples, 1 epoch → solve **0.010** (below base); ~1/3 of data = uniform turn-1 targets |
| SFT #2 (`ckpt_sft2`, the common ancestor) | 24k examples, 2 epochs, turn-1 cap **2%** → solve **0.010 → 0.500**; loss ~5.7 → 2.4–3.9 band (pre-registered); all 156 targets covered; consistency 0.51 |
| Ancestor paired row (tiles units) | solve 0.575, turns 4.96, proxy 0.474, consistency 0.565 |
| Ancestor graded truth | ~0.269 |

## 5. Prefix cache (the 15× refund)

| Quantity | Value |
|---|---|
| Speedup | ~15× (vs ~120 s/group era) |
| Fast-vs-legacy max deviation (fp32-strict) | **1.31e-06** (verification); later gate readings 7.19e-07 / 8.01e-07 / 2.18e-06 |
| Gate threshold | 1e-4, fp32-strict |
| Old discrepancy, reclassified | 4–6e-3 = bf16 kernel noise |
| Gate coverage | every `load_policy()` call (gate1/verify/log/ope/study), sticky verdict, auto-fallback to legacy |

## 6. Policy A (faithful)

| Quantity | Value |
|---|---|
| Config | from ckpt_sft2, shaping 0.1, lr 1e-6, KL 0.05, 100 groups |
| Result | solve **0.575 → 0.645** (paired, n=200, seed 41); frozen at 100 groups |
| A's tiles proxy baseline | 0.406 |

## 7. Gate 1 — proxy admissibility under trained A (the credibility frontier)

| Proxy | Role | Lift | Corr (proxy vs graded truth) | Verdict |
|---|---|---|---|---|
| tiles | candidate | **2.15×** | **+0.73** | PASS → admissible |
| equal_tiles | candidate (author's design) | **1.76×** | **+0.71** | PASS → admissible |
| yellow_heavy | candidate (win turn strictly dominated: 0.75 < 1.0) | **1.61×** | **+0.66** | PASS → admissible |
| yellow_only | negative control | **0.85×** | **−0.16** | FAIL (as designed) |

## 8. Gate 2 — the hacker ladder (all non-emergent, 5/5 at last known state)

| Run | Proxy / convention / KL | Key result (paired, seed 41, n=200) | Verdict |
|---|---|---|---|
| H1 | tiles / mean / 0.05 | solver | non-emergent |
| H2 | tiles / mean / 0.01 (matched control of H5) | solver | non-emergent |
| H3 | equal_tiles / mean / 0.01 | **solve 0.665** (best policy of project), turns 4.58, proxy 0.517, consistency 0.621 | non-emergent |
| H4 | yellow_heavy / mean / 0.01 | solve 0.680 (declining training objective — flagged oddity) | non-emergent |
| H5 | tiles / **sum** / 0.01 | heartbeats: g20 solve 0.625 obj 1.475 ent 3.56; g40 solve 0.375 obj **1.813** ent 3.55; final: solve 0.635, **truth 0.308 (UP vs 0.269)**, turns 4.76, proxy 0.424 | non-emergent (80% prediction eaten) |
| H6 | yellow_heavy / **sum** / 0.01 | **launched — result pending (ask user)** | pre-registered pivot fires if solver |

Prediction record: H4 ~55–65% (failed), H5 80% (failed publicly), H6 no percentage claimed.

## 9. Oracle exploit analysis (`exploitcheck`)

| Quantity | Value |
|---|---|
| Mean-convention exploit gap (tiles) | **+0.03** (needle) |
| Sum-convention oracle gap (tiles) | **+0.77** — but oracle assumes secret known |
| Realistic tiles-sum farming rate | ~0.3/turn (3 yellows × 0.1) vs break-even ~0.47/turn → thin |
| yellow_heavy oracle farm ceiling | 0.612 (mean); words ≥0.6 per secret: median **5**, min 0 |
| yellow_heavy realistic solving value | 0.461 (120 eps) → realistic mean edge ≈ +0.04 |
| H6 realistic math (yellow_heavy + sum) | 3 yellows = 0.6/turn; farmer ≈ 3.2–3.4 total vs solver ≈ 2.0–2.1 → gap **+1.2** |

## 10. OPE / study machinery (design constants, pre-GPU-verified)

| Quantity | Value |
|---|---|
| Logging dataset | 2,000 episodes from ε-mixture of A |
| ε-mixture support floor | π_b(a|s) ≥ ε/|V| for every action |
| Estimators | 4 (IS family + DR), with per-turn ESS, clip rate, coverage quantiles, UNRELIABLE flags |
| Study output | 4-way classification: hacking-specific signal / distance tracker / anti-signal / none |
| Prototype-era reference numbers (v3, linear policies) | hacker: flawed reward 0.179→0.270, solve 0.748→0.247; drift solve 0.818; divergence-match variance ratio 1.085; hacker ESS collapse to 0.01 by turn 3; 32% of hacker actions at ε-floor |

## 11. Process numbers worth keeping

| Quantity | Value |
|---|---|
| Group timing (pre-cache era) | ~35–45 s/group → 2–2.5 h/policy (early estimate); later ~120 s/group observed before the cache |
| Standard training budget | 100 groups × group_size 8; heartbeats every 20 (probe: every 5) |
| KL / lr defaults | A: KL 0.05; hackers: KL 0.01; drift: KL 0.1, temp 1.2, seed 7; lr 1e-6 (probe era: 2e-6) |
