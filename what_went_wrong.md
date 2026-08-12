# What Went Wrong: A Chronicle of the GPU Phase
*(Working notes for the post — every number below is from saved logs: three
FAILED training logs, smoke outputs, and heartbeats. Nothing reconstructed
from memory.)*

## The shape of the story

One evening on a Colab A100 produced **four out-of-memory crashes and three
training collapses with three distinct failure signatures** — and the third
signature is the scientifically interesting one (see the highlighted section).
The arc: memory bugs first (hardware problems CPU tests cannot see), then
learning-dynamics failures (problems smoke tests cannot see), each fix
instrumented so the next failure would be *more* informative than the last.

---

## Part I — The memory failures (OOM #1–#4)

**OOM #1 — the full-vocabulary softmax + the accumulated graph.**
First real training launch died at 18→40 GB. Two compounding causes:
(a) `log_softmax` over the model's full output tensor — batch 64 × ~110
positions × ~152k vocab ≈ **4.3 GB per chunk in fp32** — to read out ~200
numbers; (b) the group loss accumulated ~32 states' autograd graphs before a
single `backward()`, so ~32 full forward graphs lived simultaneously.
*Fixes:* window-slice logits to the word-token positions before the softmax
(~25× smaller), backward per state (mathematically identical gradients, one
graph alive at a time).
*Lesson:* the CPU self-test suite and mock pipeline verified **logic**, not
memory — memory behavior is a property of hardware execution. This birthed
the `smoke` command: one mini training group + peak-CUDA-memory print,
mandatory after any change to dtype, batching, or autograd structure.

**OOM #2 — the fp32 overcorrection.**
After the bf16 hypothesis (Part II), training was switched to full fp32 —
which fixed the weights and **doubled every activation** on top of tripled
optimizer state (~38 GB). *Fix:* the industry-standard recipe derived under
fire — **mixed precision**: fp32 master weights (micro-updates survive) +
bf16 autocast compute (activations stay cheap).
*Lesson learned twice, paid twice:* both OOM #1 and #2 launched **without**
running the smoke test that existed precisely for this. Rule since: no
exceptions, including when the person proposing the change is sure.

**OOM #3 — the workload-dependent peak.**
Smoke passed at 35.1 GiB (flagged: "thin margin"); the real run died at group
~25. Mechanism: smoke plays short episodes, but the (partially trained) policy
played deliberate **full 6-turn games**, whose prompts are ~45% longer —
activations scale with prompt length, and the peak crept past 39 GiB. The
crash site (a reference-model call) was the last straw, not the cause.

**OOM #4 — the chunking fallacy (a wrong fix, owned).**
The attempted fix — halving gradient chunks 16→8 — OOM'd *immediately*,
exposing a wrong mental model: **autograd retains saved activations for all
156 scored sequences until the state's backward, independent of chunk size.**
Chunking shrinks only the transient working set, not the retained graph.
*Correct fix:* gradient checkpointing (recompute activations in backward):
peak **35.1 → 11.4 GiB** for ~35% more compute.
*Lesson:* know which memory is transient and which is retained; they respond
to different levers.

**The cost lesson underneath all four:** exact action probabilities for OPE
require scoring all 156 candidates per state with gradients — a **~150×
compute multiplier** versus standard single-completion GRPO. That is the tax
on clean importance sampling; it was the right trade, under-quoted. (The
structural refund — KV-prefix caching, the ~110-token prompt computed once
instead of 156 times — is the designated fix for both cost and memory, gated
by score-equivalence asserts.)

---

## Part II — The training collapses (three signatures)

**Collapse #1 — sparse reward, the self-sealing spiral.**
Reward = solve-only (1 if win). Heartbeats: 0.125, 0.375 early, then **96
consecutive zero-solve episodes** — probability ≈ 0.9⁹⁶ ≈ 4×10⁻⁵ under the
base rate. Not "no learning": degradation *below* base.
Mechanism: at ~10% solve, ~40% of 8-episode groups are all-fail →
group-standardized advantages are all-zero → **zero gradient** (GRPO needs
within-group reward variance); in mixed groups, 7 losers' negative advantages
punish every action they took — mostly the sensible head words — far more
often than the single winner reinforces them. Head erodes → fewer wins →
less signal: self-sealing.
*Fix attempt:* dense truth-aligned shaping (per-turn feedback-consistency,
strictly dominated by the solve reward) — which exposed a hidden bug:
`consistency_q` was a hardcoded `0.0` placeholder in training turns, so the
shaping flag would have been silently inert. Plus: heartbeat split into raw
`solve` vs shaped `objective`, so a gamed shaping signal would be visible.

**Collapse #2 — dense reward, and the bf16 red herring.**
Objective fell 0.258 → 0.089 → floor. Same shape as #1 under a *different*
reward pointed at the optimizer. Suspect: bf16 master weights truncate AdamW
micro-updates — at lr 2e-6 the step is ~40× below the bf16 ulp of typical
weights (~8×10⁻⁵), so most updates round to zero and survivors apply as
biased noise. The physics is real and the fp32-masters fix is correct
practice — but the probe later showed it was not the disease here. A red
herring worth keeping in the post: a true mechanism that didn't explain the
observation, identified as such by a controlled single-variable probe.

**Collapse #3 — the probe verdict, and the signature that matters.**
Configuration: fp32 masters + bf16 autocast, dense shaping, checkpointing,
plus a new instrument — per-group sampling entropy (proposed by the author
after asking "could the model lock onto early exploration and never come
back?"). Result: objective 0.154 → 0.075 (floor) while entropy rose 2.69 →
4.42 toward uniform (max ln 156 = 5.05). One probe, two hypotheses killed:
not numerics (fp32 in, decline persists), and not premature-convergence
lock-in (that predicts entropy → 0; it did the opposite). Third signature:
**prior erosion / flattening** — net negative advantage pressure from ~7/8
failing episodes grinds down the frequently sampled head words faster than
rare wins rebuild them, and the policy unlearns its priors toward uniform.
Diagnosis-dependent cure, worth stating: an entropy bonus — the reflex fix
for RL instability — would have made this strictly worse. The cure: SFT warm
start. Distill the consistency heuristic (0.98 solve) into the base model;
initialize A, B_hack, and B_drift from this one common ancestor (differences
downstream attributable to reward alone); KL now anchors to a competent
reference. This is the field's standard recipe — which is exactly the point
of the highlighted section below.

---

## ★ THE SECTION THAT EARNS ITS PLACE: entropy *inflation*, not collapse ★

The GRPO pathology the 2025 literature documents is **entropy collapse** —
policies becoming overconfident, exploration dying, and a body of work on
entropy mechanisms in RLVR built around preventing exactly that. **We
observed the opposite sign: entropy inflation — prior erosion toward
uniform — in the low-success regime (base solve ~10%), with a small model
(0.5B), from scratch, over a constrained 156-way action space.**

Why hasn't this regime been characterized? Because **everyone warm-starts.**
DeepSeekMath ran GRPO on an instruction-tuned model; DeepSeek-R1 inserted
cold-start SFT before RL and reported that for *small* models, distillation
beats direct RL; InstructGPT fixed the SFT→RL ordering in 2022. The
from-scratch, low-success, small-model corner is empty not because it's
uninteresting but because the field learned to route around it before
instrumenting it.

The pieces of this observation and where they sit against the literature:
the **zero-gradient all-fail groups** are documented (DAPO's dynamic sampling
exists to filter exactly them); the **conclusion** — small models need SFT
before RL — is documented (R1's distillation finding); what appears
under-documented is the **observable signature in between**: objective
pinned at floor *while sampling entropy rises*, i.e., the failure is not the
policy freezing but the policy *dissolving*, driven by negative-advantage
pressure on the sampled head. The instrument is one line (mean sampling
entropy per group) and the testbed reproduces the effect in under an hour of
A100 time, from a public repo, with saved logs.

Stated with the honesty it needs: this is a **single-seed, single-model,
single-environment empirical observation on a non-standard policy class**
(softmax over full-sequence logprobs across a closed 156-word action space —
not token-level GRPO), offered as a documented signature plus a mechanism
hypothesis, positioned between DAPO (the gradient-starvation half) and R1
(the warm-start conclusion). It is not a theorem and not a benchmark claim.
But it is exactly the kind of small, controlled, reproducible observation
the open-RLVR conversation runs on — and it fell out of a debugging session
because the diagnostic (entropy) was added *before* the run, to test a
hypothesis, rather than after, to explain a corpse.

---

## Cross-cutting lessons (the post's "methodology" thread)

1. **Verification is layered, and each layer is blind to the next.** CPU
   self-tests catch logic; the mock pipeline catches integration; GPU smoke
   catches memory; only real training catches learning dynamics. Every
   failure tonight occurred precisely one layer above where testing stopped.
2. **Instrument before interpreting.** The solve/objective split and the
   entropy heartbeat were added to test *named hypotheses*; both paid off by
   making the third failure diagnostic instead of mysterious.
3. **Failure logs are data.** Three FAILED logs were saved deliberately; they
   are the figures of this section.
4. **Probes over full runs; verdicts as early as the data allows.** The
   60-group probe was sized from the observed failure timescales (both
   collapses declared themselves inside 40 groups) and called at group 40.
5. **Guardrails catch their author.** The assert-guarded patcher and the
   memory-pattern tripwires fired three times on the code's own author —
   including once on the tripwire's own source, and once as a documented
   false positive on legitimate single-graph accumulation.
6. **Being wrong fast is the methodology.** Four confident diagnoses were
   issued tonight; two were wrong (chunking, bf16-as-the-disease) and both
   were falsified within one cheap experiment each. The system's value is
   not being right — it is making wrongness cheap, visible, and short-lived.

---

## Coda — the cure failed once too, and the arc it completes

The SFT warm start did not work on the first attempt either, and the way it
failed is what ties the whole night together.

**SFT #1 (3k examples, 1 epoch): solve = 0.010 — below the base model.** The
mechanism was hiding in the teacher's own mathematics: on an empty history
every candidate word is consistent, so the consistency-softmax teacher is
exactly uniform on turn 1 — at any temperature, since softmax of equal
scores is uniform regardless. Roughly a third of the dataset therefore
taught (no guesses yet) -> random word, and cross-entropy toward a uniform
target does not get diluted by data — it gets learned. The supervision
flattened the base model's decent opener prior (apple / quick / peace,
measured before any training) faster than the sparse mid-game examples could
install the feedback logic. Scaling the dataset without fixing this would
have made it strictly worse: 8x more gradient teaching the flattening.

The fix was data quality, not scale alone: cap empty-history examples at 2%
of the dataset — deliberately under-training turn 1 so the base model's own
opener prior survives untouched (preserving a good prior beats teaching any
replacement) — plus dataset diagnostics printed before any GPU time burns
(empty-history fraction, turn-index distribution, teacher solve rate, target
diversity). Two proposed fixes were rejected on analysis rather than taste:
lowering the teacher temperature is a mathematical no-op on equal scores,
and argmax-over-uniform is index-order arbitrariness that would also have
shrunk within-group variance for later GRPO.

**SFT #2 (24k examples, turn-1 capped, 2 epochs): solve = 0.010 → 0.500.**
Loss descended from ~5.7 to the 2.4–3.9 band exactly as pre-registered
("well below 4, or the data is exonerated and capacity/lr indicted"), all
156 target words covered, consistency 0.51 — feedback logic partially
installed, which is precisely what SFT #1 never achieved. And 0.5 solve is a
near-ideal GRPO starting point: maximal within-group reward variance, where
group-relative advantages are strongest and the erosion dynamic cannot take
hold.

**The completed arc.** Across one night, the same underlying failure — a
useful prior being flattened — appeared through three unrelated mechanisms:
RL negative-advantage pressure grinding down the sampled head (collapse #1
and #3), bf16 update truncation applying biased noise to the weights (the
red-herring physics of collapse #2), and uniform imitation targets teaching
flatness by supervision (SFT #1). Three diseases, one symptom, and one
instrument that saw them all: watching the policy's entropy and its
objective together, so that "getting worse" could be told apart from
"getting uniform," "getting stuck," and "getting noisy." The generalizable
lesson is not any single fix — it is that the diagnostic was cheap, was
added before the failures rather than after, and turned every one of them
from a mystery into a mechanism.

---
---

# SINCE LAST TIME — Part III onward
*(Everything below was crossed after the SFT #2 coda. Same rule: every
number is from saved logs — training logs, gate outputs, paired eval
tables, and the exploitcheck oracle tool. Nothing reconstructed from
memory.)*

## Part III — The prefix-cache era: buying back the 150× tax

The structural refund promised in Part I was cashed. **KV-prefix caching**
computes the ~110-token prompt once per state and scores all 156 candidate
words as short continuations — the designated fix for both the compute tax
and the memory pressure.

**The equivalence archaeology.** Fast-vs-legacy scoring initially disagreed
by 4–6e-3 — alarming for a pipeline whose importance ratios are literal
quotients of these numbers. The archaeology ended on one number:
**1.31e-06 maximum deviation under fp32-strict comparison**, on both policy
instances. The implementation was mathematically correct all along; the
4–6e-3 was **bf16 kernel noise**, reclassified from gate-failing to
runtime-acceptable. Observed gate readings across later runs: 7.19e-07,
8.01e-07, 2.18e-06 — all far under the 1e-4 fp32-strict threshold.

**The gate became load-bearing.** After a review catch (the author's): the
prefix-cache equivalence gate initially ran only in `train` and `smoke`,
while `gate1`, `verify`, `log`, `ope`, and `study` — the *most*
probability-sensitive commands — would have used the unverified fast path.
Fix: `load_policy()` verifies the cache unconditionally at **every** policy
load, verdict is sticky, and any fast-path failure (gate or mid-run
exception) degrades loudly to the GPU-proven legacy scorer instead of
crashing or silently corrupting probabilities. Promised failure mode: "no
speedup," never "wrong numbers."

**The payoff.** Smoke peak with checkpointing + prefix cache: **12.0 GiB**
on the 40 GB card (from the 35.1 GiB era) — memory solved twice over — and
roughly a **15× speedup**, collapsing ~120 s/group budgets and unlocking
"possibly all of it today" cadence.

**The lesson, owned:** three band-aid patches in a row had fought the same
root cause before the rewrite happened; cumulatively they cost more than the
rewrite. *When three patches in a row fight the same root cause, stop
patching and fix the root.* The right moment was the author's "5 hours for a
simple game?!" message.

## Part IV — The warm start pays: A trains, and the collapse diagnosis is confirmed

GRPO from `ckpt_sft2` (solve reward + 0.1 truth-aligned shaping, lr 1e-6,
KL 0.05, 100 groups) produced the run the whole chronicle had been building
toward: **solve 0.575 → 0.645** (paired, n=200, seed 41), entropy stable,
no erosion. The direct test of Part II's diagnosis passed: with a competent
KL anchor and maximal within-group reward variance, the erosion dynamic
never took hold. **A was frozen at 100 groups on purpose** — a stronger A
would only raise the proxy baseline that Gate 2's hacker must exceed,
making the experiment harder while proving nothing new.

## Part V — Gate 1 grows teeth: the credibility frontier

Gate 1 rerun under the *trained* A (the proper instrument, replacing the CPU
heuristic) produced a monotone credibility frontier along the
yellow-weighting axis — itself a post figure:

- `tiles` — lift **2.15×**, corr **+0.73** → **admissible**
- `equal_tiles` (author's design, removes the solver premium) — **1.76×**,
  **+0.71** → **admissible**
- `yellow_heavy` (Y=0.2, G=0.15 per tile; the first proxy whose win turn is
  *strictly dominated*) — **1.61×**, **+0.66** → **admissible**
- `yellow_only` (designated negative control) — **0.85×**, **−0.16** →
  **FAIL**, exactly as designed; code asserts it can never be trained on.

The gate's rejection boundary is visible between the last two rows: a gate
that only ever says yes proves nothing; this one shipped with something to
say no to — and said it.

## Part VI — The ladder of non-emergence: five hackers that refused to hack

Five attempts to make reward hacking *emerge* from optimization against
Gate-1-admissible proxies, all from the same ancestor, all certified
honestly, all **non-emergent**:

- **H1** (tiles, mean, KL 0.05): non-emergent.
- **H2** (tiles, mean, KL 0.01): non-emergent — and, by the single-lever
  audit, the *only* matched control for the convention ablation.
- **H3** (equal_tiles, mean, KL 0.01): **the best policy of the entire
  project** — solve 0.665 (above A's 0.645), turns 4.58, higher
  consistency. A "hacker" training run produced the project's top solver.
- **H4** (yellow_heavy, mean, KL 0.01): non-emergent; paired solve 0.680
  with a *declining* training objective — flagged as a seed-luck oddity for
  a later second-seed re-eval, not blocking.
- **H5** (tiles, **sum** convention, KL 0.01): heartbeats objective 1.475 →
  1.813 by group 40 (the fastest proxy-climbing of any run) — and the final
  row was still a solver: **solve 0.635, graded truth 0.308 (ancestor
  ~0.269, i.e. UP), turns 4.76 (faster), proxy 0.424.** The pre-registered
  80% emergence prediction was **eaten publicly, as promised: five for
  five.**

**The instrument fix that made H5 readable (pre-registered before its
data existed):** the project had run with a latent inconsistency — Gate 1
measured proxy credibility against *graded* truth while Gate 2 certified
emergence against *binary* solve. The sum-exploit's predicted signature
(solve late, solve long, still solve) is invisible to binary solve by
construction. Gate 2 was upgraded to **graded truth** (solve speed) as the
verdict channel, with binary solve reported alongside — and the criterion
change was audited as **verdict-preserving on H1–H4** (the anti-goalposting
receipt). Division of labor, stated once and kept: *sum-vs-mean is the
treatment; graded truth is the detector.* One changes the world; the other
changes only the referee's eyesight.

## Part VII — The oracle explains the graveyard

Rather than a sixth blind run, an exhaustive `exploitcheck` oracle tool
computed the exploit's actual worth:

- **Mean convention:** exploit gap ≈ **+0.03** — a needle. Under
  episodes-end-on-solve, the winning turn is the best-scoring turn for every
  monotone letter-credit proxy, and ending early prunes weak turns from the
  mean. Fast solving is near-optimal for the whole proxy family.
- **Sum convention (tiles):** oracle gap **+0.77** — but the oracle assumes
  the secret is known. A realistic farmer holds discovered letters as
  yellows at 0.1 each (three yellows = 0.3/turn), *below* the ~0.47/turn
  break-even where stretching beats ending. The realistic gap was thin all
  along: the 80% was built on the oracle number with the caveat
  under-applied — the quantitative post-mortem of the eaten prediction.
- **yellow_heavy:** oracle farming ceiling 0.612 vs realistic solving value
  0.461; median only **5 words per secret** score ≥ 0.6 (some secrets:
  zero). Realistic edge ≈ **+0.04** under mean.

One theory now covers all five non-emergences: *on-distribution credibility
means the local gradient points at quality; the diverging exploit is a
combinatorially precise strategy worth pennies, invisible to noisy
group-relative gradients under a KL anchor.* The accidental finding worth
quoting: **KL anchoring + mean-convention hygiene + episodes-ending-on-solve
form a defense stack strong enough that two different flawed proxies both
produced our best solvers.**

## Part VIII — Rung six and the pre-registered pivot

**H6 = yellow_heavy + sum** — the two-lever combination, licensed by the
single-lever discipline only after two documented one-lever failures (H4:
weights-only; H5: convention-only), each thin for the *same* diagnosed
reason. The realistic (non-oracle) math finally clears the bar: three held
yellows pay 0.6/turn under yellow_heavy, every farming turn is additive
under the sum — farmer ≈ 3.2–3.4 total vs fast solver ≈ 2.0–2.1, a **+1.2
gap built from the cheap resource**. No prediction percentage this time;
calibration on emergence was publicly demolished, so the run gets the math
and nothing more. If H6 emerges, attribution is still clean: *the
interaction* did it — "neither misaligned weights nor naive aggregation
sufficed alone; together they did."

**And the pivot was registered before H6's outcome existed:** if rung six
also produces a solver, the ladder ends and the paper splits into two honest
halves. **Part one, already banked: emergence resistance** — six attempts,
the oracle analysis, the accidental defense stack. **Part two, the actual
research question — detection** — proceeds by distilling a scripted farmer
teacher into Qwen via the existing SFT pipeline: a genuine learned LLM
policy with real logprobs whose hackedness is certified on-policy by
construction, because detection-from-logs is meaningful regardless of how
the hacked policy came to exist. That decoupling unblocks the OPE study
(log → ope → study) instead of laddering forever.

## New cross-cutting lessons (appended to the methodology thread)

7. **Verify the instrument at every consumption site, not just where it was
   born.** The prefix-cache gate guarded training but not the
   more-sensitive OPE path until a review caught it; now every policy load
   verifies, with loud fallback. Speed must never silently change the
   science.
8. **Oracle computations before blind runs.** One exhaustive CPU
   computation (`exploitcheck`) explained four GPU non-emergences and
   correctly priced the fifth and sixth rungs — cheaper than any training
   run, and it converted "why won't it hack?" from mystery to arithmetic.
9. **Match the pair before claiming the ablation.** "Four mean runs + one
   sum run" is not a convention ablation; H2-vs-H5 (one variable flipped)
   is. The author's audit demoted H3/H4 to the proxy-weight axis where they
   belong.
10. **Pre-register the escalation, the framing, and the pivot — then let
    the gates spend them.** Every rung of the ladder, the graded-truth
    upgrade, and the split-paper pivot were all written down before the
    data that would judge them existed. Non-emergence reported as
    non-emergence, five (soon six) times, is what makes the eventual
    positive — from whichever half of the paper — believable.
11. **Eat wrong predictions in public, with the quantitative post-mortem
    attached.** The 80% failed because an oracle number was used where a
    realistic number belonged; saying exactly that is worth more to the
    post than the prediction succeeding would have been.

## Coda II — one symptom, one arc, continued

Part I–II's arc was "a useful prior being flattened" seen through three
mechanisms. The new arc is its mirror: **a useful prior refusing to be
corrupted** — five optimizers offered five admissible flawed rewards, and
every one of them climbed back to alignment, because the defense stack made
honesty the path of least resistance. Whether H6 breaks the pattern or the
pivot fires, the chronicle's through-line holds: cheap instruments, added
before the data, pre-registered rules, and failures promoted to findings.

---
---

# THE FINAL STRETCH — Parts IX–XIII
*(July 6–7. Same rule as always: every number below is from saved logs —
train_H6/H7.log, bon.log, farmer*.log, train_D*.log, verify*.log,
A_logs.jsonl, study_results*.json. Nothing reconstructed from memory.)*

## Part IX — The author demands research, and the research changes the plan

After five non-emergences the author stopped the engineering and issued the
correct rebuke: *"you are very efficient to find thousand papers to show me
that already exists when I propose a novelty, but very bad to find in paper
solutions."* The literature review that should have preceded the ladder was
done after rung five — and it recast everything:

1. **The KL coefficient is the field's anti-hacking brake.** The canonical
   overoptimization study ran its main experiments at KL **zero** and found
   the penalty acts like early stopping; overoptimization grows with distance
   traveled from init. Our hack runs used 0.05 and 0.01 — never 0.
2. **The exploit must be sampled before it can be reinforced.** A GRPO
   gridworlds study found the mirror image of our situation: exploit-basin
   initialization → hacks immediately, solution never sampled. Same
   algorithm, opposite basin.
3. **Best-of-n over-optimizes at far lower KL than RL** — selection amplifies
   exactly the tail trajectories on-policy gradients never see.
4. The mitigation recipe the field recommends — SFT anchor + KL — is
   **exactly the stack we had accidentally assembled.** The thesis was
   already in our own logs.

The cheap instrument this spawned: a **discovery probe** on the ancestor
(200 episodes, CPU-adjacent). Result: corr(length, sum_yellow_heavy) =
**+0.47**, with 6-turn episodes collecting 2.00 total against 1.06 for
2-turn solves. The anti-solving gradient was *in-distribution from group
one* — exploration was never the bottleneck. One pre-registered prediction
(positive correlation) landed; the emergence estimate moved 50% → 65%.

*Process lesson, owned in both directions:* the model's search behavior had
been triggered by novelty-checking and never by engineering plans; nobody —
author or assistant — had asked "will this configuration produce hacking?"
as a literature question. **Literature review is a gate too. Run it before
the ladder, not after rung five.**

## Part X — Rungs six and seven close the ladder

**H6 (yellow_heavy + sum, KL 0.01)** got a new one-line instrument first —
mean turns in the heartbeat — and the instrument earned its place
immediately. Mid-training, the first farming signature of the entire
ladder: groups 60–80 climbed to objective **2.750** with turns **6.00**.
By group 100 it was back to 3.75 turns, and the paired verdict was flat
everywhere: proxy 0.407→0.405, truth 0.297→0.287, solve 0.605→0.580 —
**non-emergence #6**, of a new kind: not "trained into a better solver"
but *nothing moved net of the excursion*. Something dragged the policy back
to the anchor, and only one term in the loss drags toward the anchor.
The author's observation — "at step 80 it was bad and 100 good" — was the
right catch wrongly mechanized (single-secret heartbeat noise forbids the
step-level reading, and no, the model does not "learn to use the negative"),
and it produced a permanent instrument: **trajectory snapshots every 20
groups** (to VM-local disk after Drive quota reality intervened), because
proxy-vs-truth trajectories are non-monotonic and the hacked policy may
exist only mid-training.

**H7 (identical, KL = 0)** was the literature's prime suspect isolated in
one lever. Verdict: proxy 0.407→**0.403**, truth 0.297→0.278, solve
0.605→0.605. **Non-emergence #7 — and the KL coefficient acquitted as the
sole cause.** With the leash off and a proven in-distribution gradient, the
policy did not move *even on its own training objective*: at lr 1e-6 × 100
groups × group-size 8, this optimizer simply does not travel. (A moved
0.575→0.645 on the same budget — but with dense shaping; the pure-proxy
group-relative signal is evidently too weak.) A footnote for honesty: the
snapshot patch never reached the running copy — the stale-copy trap's third
and costliest bite — so H7's mid-training checkpoints are lost; the
heartbeats (oscillation, no sustained march) put low odds on a hidden farmer.

**The ladder closed at 7/7, exactly as pre-registered.** Part 1 of the paper
is banked with a stronger closing line than 5/5 had: *the defense stack held
even with the KL term removed.*

## Part XI — Manufacturing the hacker: two failures the gates caught, then the pass

The pivot ran as an escalation of manufacturing methods, and Gate 2 rejected
two defective products before accepting one:

**BoN distillation, round 1** (sample 12/secret at temp 1.3, keep top-3,
SFT on winners): selection lift 1.30×, selected episodes 5.74 turns — the
signal existed. But the product was **degradation, not hacking**: truth
crashed (0.292→0.181) *and the proxy fell with it* (0.411→**0.366**).
Distilling temp-1.3 samples cloned noise; SFT loss never left the 3.6–4.9
band. The gate's FAIL protected the study from certifying exactly the
drift-confound shape it exists to distinguish. **Round 2 self-diagnosed in
its first prints** — selected-sum *below* round 1's with length already
saturated at 5.90/6: selection exhausted on a degraded pool — and was
correctly aborted at secrets 40/120.

**Scripted farmer, v1** (expected-proxy teacher, temp 0.05): the student
learned exactly half the teacher — the wrong half first. Solve-avoidance is
a sharp 6-logit signal and was learned thoroughly (consistency 0.558→0.337,
solve →0.213); yellow-seeking lives in expected-value gaps of 0.01–0.03,
which at temp 0.05 are ~1.5× softmax ratios — **near-uniform targets** —
and was not learned at all (proxy 0.409, missing the bar by **0.002**;
loss floor ~4 nats was the target diffuseness itself). A quantitative
post-mortem also landed on the assistant's ledger: a "99%" success estimate
had ignored the project's *own* exploitcheck warning that the yellow_heavy
mean-convention margin is thin. Assuming a ceiling instead of measuring it.

**The fix was a gate, of course:** the **teacher self-check** — measure the
scripted teacher's own mean_proxy_return on CPU *before any GPU time*, hard
floor at 0.44. **Farmer v2** (teacher temp 0.01): self-check **0.568** —
a +0.16 ceiling over the ancestor, forty times v1's miss — then a clean
descent to the 0.6–1.5 nat band, and the first Gate-2 PASS of the project:
**proxy 0.411→0.518, truth 0.292→0.058, solve 0.147** (re-certified at
n=200: 0.521 / 0.047 / 0.125). `ckpt_HF` exists. Ten runs, one hacker —
distilled, and the Gate-2 PASS message was reworded the same hour so the
log can never claim "emerged" for it: *certification is behavioral;
emergence is a provenance claim.*

**HF-lite** (4k examples, 1 epoch) followed for the dose-response: proxy
0.442, truth 0.101, solve 0.240 — **hacked at half strength**, the
instrument Part XIII would need.

**B_drift** rounded out the cast the honest way: temp 1.2 **failed** its
own gate by three episodes (0.525 vs required 0.540 — a strict gate erring
toward protecting the control), and temp 1.1 passed emphatically: solve
**0.640** against A's 0.635, a genuinely benign twin.

## Part XII — The ground moves: the environment becomes a gated variable

Between the July 6 and July 7 sessions, Colab's image advanced the
transformers library — which had switched to **weekly releases that ship
breaking changes** (their own migration guide says to pin your exact
version) — to 5.12.1, and the stack died on its first forward pass. The
debugging descended through two false floors: embedding receiving *float*
indices (patched with 11 explicit `dtype=torch.long` pins — correct
defensive practice, wrong diagnosis), then a sequence-length-**zero**
reshape, which finally named the real bug: **tokenizer files saved into
checkpoints by the older version parse without error under 5.11+ and encode
every string to zero tokens.** Empty inputs → float empty tensors → both
crashes, one cause.

Two fixes, one permanent lesson. The **tokenizer gate**: every policy load
probe-encodes; on empty output it falls back to the hub tokenizer — *exact*,
not approximate, because the design already requires all policies to share
one tokenizer for IS validity — and self-heals future checkpoints by saving
the working format. And the **pin**: `transformers==5.11.0` in
requirements-gpu.txt and a mandatory session preamble whose second cell
prints the versions before anything spends a unit. Recovery certificate:
cache gate PASS at the familiar 1.48e-06, all self-tests green.

*The lesson, stated the way the OOMs were:* **an unpinned dependency is an
unregistered experiment variable.** It "worked all this time" the way an
unseeded RNG works — until it doesn't, silently, between sessions. The same
discipline that fingerprints the code now pins the ground it stands on.

## Part XIII — The endgame: the study runs, and the answer is structural

2,000 episodes logged from ε-wrapped A. Then the tables the project exists
to produce — twice, at two hackedness doses.

**What worked, cleanly.** Inside coverage, the estimators were excellent:
drift estimated **0.397 vs 0.401 true** (ESS 0.64); H6 — recycled from the
ladder as a bonus *proxy-pressured-but-non-hacked* control — **0.396 vs
0.382** (ESS 0.48), with **no false alarm** on any diagnostic. And outside
coverage the machinery *knew it*: HF's row printed ESS 0.00, max weight
223, **UNRELIABLE**, instead of a fake number.

**What the question turned out to be.** To match HF's weight-variance, the
divergence-matcher had to push drift to temp 2.0–2.2 — where drift **stops
being benign** (solve 0.24–0.28). *Within the matcher's search family
(temperature-scaled drift), no benign policy exists at the hacker's
distance* — every candidate far enough to match had ceased to be benign.
Every separating diagnostic (snis_mpr sAUC 0.98,
top1pct_mass 0.89) was therefore correctly classed **distance tracker /
TRAP** by the study's own pre-registered classifier; var and maxw tracked
raw divergence at corr +1.00. Outcome 2, reported as Outcome 2.

**The dose-response made it structural.** HF-lite — half the hackedness by
every behavioral measure — collapsed coverage *identically* (ESS 0.00, max
weight 2041, matching failed at the grid maximum with var_ratio 0.06). Two
doses, one collapse, and the mechanism is arithmetic: passing Gate 2
*requires* systematic multi-turn deviation from the solve-trained logger;
IS weights compound multiplicatively per turn and die under exactly that.
**In this regime, "hacked enough to certify" and "close enough to estimate"
exclude each other.**

**The severity axis, closed (the author's catch, one day before writing).**
Both hackers so far were truth-*destroying* (solve 0.13, 0.24) — a skeptic
could dismiss the exclusion as "of course a broken policy that stopped
playing leaves coverage." So a third hacker was built to answer it: **HL**,
a *late-solve farmer* (`solve_after=4`: farm four turns, then solve). It
passed Gate 2 while **solving MORE than the honest policy** — solve
0.613→**0.767**, graded truth 0.292→**0.241**, proxy →0.547: a hack that
*raises the naive binary-solve monitor* and is only visible as a hack on
the efficiency channel. The realistic threat model. And it collapsed
coverage too — ESS 0.00 — but with a telling difference: **max weight
15.3, not HF's 223 or HF-lite's 2041.** HL's per-action deviation is mild
(farming with consistent words overlaps A's support); it is *close*, far
closer than the others — and four farming turns still compound that mild
per-turn bias into zero effective sample size. That sharpened the
mechanism from "hackers are grossly off-distribution" to **"any systematic
per-turn bias, however small, compounds over the horizon into coverage
collapse."** Three distinct hackedness profiles, one verdict.

**[RETRACTION NOTE, added later — see Part XV.]** HL's Gate-2
certification did not survive replication: the sa=4 recipe, run three
times, measured truth deltas of −0.051 / −0.002 / +0.002 — a truth-NEUTRAL
family whose one lucky draw a noisy gate certified (winner's curse). The
*measurements* in this section (ESS 0.00, maxw 15.3) were real; the
"hacked" label was not. The solve-preserving endpoint of the severity axis
was subsequently re-established honestly by **HL5** (fractional
`solve_after=4.5`), certified on a paired instrument at n=200 — see Parts
XIV–XVI. The mechanism sentence survives the retraction because it is
behavioral: sa=4-family policies farm identically whether or not they
damage truth, and HL5 (maxw 63.9) carries the mild-deviation-still-dies
claim with a valid label. From logs, hacking is unmissable as anomaly — and
unattributable as hacking *from these logs, by these diagnostics*: its
entire signature is the coverage collapse other severe shifts also leave.
The in-house witness for that conflation is **BoN r1** — far from A,
truth-destroyed, and *not hacked* (its proxy fell too); a distance alarm
would flag it identically to the hackers. It never entered the study only
because Gate 2 rejected it as an input — which is the point.
*(CORRECTION, Part XVII: this "would flag it identically" was an
assertion, not a row — and when the row was finally measured on Aug 5, it
was WRONG: BON1 prints ok, ESS 0.12, estimate accurate. The witness role
passed to IMP, measured. The paragraph is preserved as written because
the error's survival across drafts is itself Part XVII's lesson.)*

**The positive result inside the negative (the author's reframing, caught
while red-teaming the claims before writing).** State what the alarm *is*
rather than what it isn't: over the benign class this study defined —
same-ancestor retraining with one knob varied — the coverage alarm was a
**sound not-benign tripwire**: zero false alarms (including on H6, the
proxy-pressured non-hacker), a guaranteed trip on every certified hacker
*with a mechanism for the guarantee* (Gate-2 hackedness necessitates
multi-turn deviation; deviation kills coverage), and self-reported
reliability. Two things it is not, stated with equal precision: it is not
a hacking detector (*the draft's evidence here was "BoN r1 trips it
without hacking" — refuted on measurement, Part XVII; the corrected
evidence is stronger: IMP, the benign super-solver, trips it without
hacking, measured*), and its benign-side
guarantee is **scoped to that class** — a benign policy of equal
competence but *different strategy* (e.g. a different opener book, taking
turn-1 actions A gives ~10⁻³) would plausibly trip it too, and none was
constructed. The temperature family structurally cannot produce one
(temperature scaling preserves A's preferences), so the matcher's search
never visited that region. A monitoring primitive with its guarantee
conditions spelled out — "silence is informative for same-lineage
one-knob updates; any alarm means *not that*, pull the policy and run
on-policy evals" — is the deployment-shaped half of the finding.

The pair of results is the paper's Part 2, and it is a finding, not a
failure: IS-family diagnostics monitor *distribution*, not *intent* —
which makes them a sound, scoped tripwire and an unsound attributor.
Attribution needs something the logs of a faithful policy structurally do
not contain about a policy that has stopped being faithful; the one
channel that could supply it (the reward-truth gap) is itself estimated
through the weights that departure destroys. (BoN-v2 as a second hacker
was withdrawn on these grounds — it would collapse coverage identically and
add a row, not a result.)

## New cross-cutting lessons (12–16)

12. **Cheap decision-relevant information before expensive optional
    enrichment — always.** The three-day trough came from serial gambling
    on the critical path; the recovery came from probes (discovery,
    teacher self-check, tokenizer probe) that cost minutes and decided
    branches before GPUs spent hours.
13. **Gates protect the study from its own inputs.** Gate 2 rejected a
    degradation-shaped BoN product and a half-learned farmer — either
    would have poisoned the detection study with a confound. A gate that
    fails the right things is worth more than one that passes.
14. **Measure the ceiling before training toward it.** The teacher
    self-check exists because a confident estimate substituted for a
    30-second measurement, once.
15. **Dose-response is the cheapest structural argument.** One extra
    half-strength hacker converted "the study failed" into "the exclusion
    is severity-independent" — the difference between an anecdote and a
    mechanism.
16. **Pin the ground.** Code fingerprints, seeds, gates — and now library
    versions. Every silent variable eventually presents its invoice.
17. **Red-team the claims like the code.** The strongest sentence of Part 2
    ("no benign policy exists at the hacker's distance") did not survive
    the author's pre-writing scrutiny — it was true only within the
    matcher's search family — and the corrected version surfaced a
    *positive* result the negative framing had hidden (the scoped
    tripwire). Fourth entry in the same ledger (H2/H5 matching, the
    Gate1/Gate2 inconsistency, equal_tiles' design): the project's
    sharpest framing corrections all came from one move — *state
    precisely what was shown.*
18. **A withdrawal is a claim too — check it.** The assistant withdrew the
    second-hacker experiment ("it would collapse coverage identically, add
    a row not a result"), and that reasoning silently assumed all Gate-2
    hackers are truth-destroying. The author had pushed the variant
    repeatedly; running it (HL, solve-preserving) both *strengthened* the
    result — the exclusion now spans the full severity range, including a
    hacker that looks healthy on solve rate — and refuted the withdrawal.
    Not doing an experiment is a decision that earns the same scrutiny as
    doing one. *(Later postscript: HL's own certification was then
    retracted under replication — Part XV — and the severity-range claim
    was re-earned by HL5. The lesson stands twice over: the experiment was
    worth running, AND its result was worth re-checking.)*

## Coda III — the arc, closed *(as it stood July 7; the true ending is Parts XIV–XVI)*

Part I–II: a useful prior being flattened, three mechanisms, one
instrument. Parts III–VIII: a useful prior refusing corruption, five
optimizers climbing back to alignment. The final stretch inverted the
question one last time: when a policy *is* finally corrupted — built to be,
because seven optimizers declined — the logs of the honest policy see
that something left the distribution, loudly, immediately, and with
self-reported reliability; over the benign family we defined, that alarm
never cried wolf and never missed a hacker. What the logs cannot say is
*why* it left. The project set out to ask whether off-policy diagnostics
detect reward hacking; it ends knowing precisely which words in that
sentence are true: **a sound, scoped not-benign tripwire, yes;
attribution of hackedness, no — not in this regime, not from these logs,
not by these diagnostics** — with every link in that answer certified by
a gate written down before the data existed, and the final framing itself
corrected by the same discipline, one day before the writing began.

---
---

# THE INSTRUMENT TURNS ON ITS OWN RESULTS — Parts XIV–XVI
*(Late July – August 4. Same rule as always: every number is from saved
logs — gate1_paired.log, farmer_latesolve2/3.log, farmer_hl4.log,
verify_HL5_paired.log, ope_HL5.log, study_HL5.log, plus the verify/sft
self-check outputs quoted verbatim in the work record. Nothing
reconstructed from memory.)*

## Part XIV — The paired instrument: fixing a seam the gates had been ruling through

**The seam, found by asking a simple question.** "Are the Gate 1 A-arm and
random-arm episodes the same games?" They were not — one rng object,
consumed sequentially, meant the two arms played disjoint secrets. Worse,
the same held for every Gate 2 comparison: `onpolicy_eval` seeded each
policy identically, but action sampling consumed the stream at different
rates, so from episode 2 onward the "paired evals" in the master table
were paired in seed-label only. The structural blocker was one line:
`run_episode` drew the env secret from the same rng that sampled actions.

**Why it mattered, quantitatively.** Gate 1 decides across a chasm (lifts
1.6–2.4× against a 1.5 threshold; unpaired noise irrelevant). Gate 2
decides on a knife edge: truth deltas of a few hundredths against an
unpaired SE of ~0.03 at n=150–200 — *the noise band was the same order as
the effect sizes the gate ruled on.* The porousness was about to be
demonstrated rather than hypothesized (Part XV).

**The fix.** `run_episode(secret_seed=)` decoupled the secret from the
action rng (backwards compatible); `gate1_final` pre-draws one seed list
and runs both arms on it; `onpolicy_eval(secret_seeds=)` plus a shared
200-secret list in `verify` makes Gate 2 and the drift gate same-secret
comparisons across ancestor/A/H/D, with a new `paired_deltas` readout
printing per-secret sign counts *under* the untouched pre-registered
mean-based criterion — the verdict rule was deliberately not changed
mid-project (pre-registration preserved; only the data got cleaner).

**Gate 1, re-run paired: all four verdicts survived the upgrade.** tiles
2.41× / +0.75 (A beats random on **95%** of shared secrets); equal_tiles
1.91× / +0.72 (93%); yellow_heavy **1.72×** / +0.67 (95% — the margin over
1.5 grew from the unpaired 1.61×); yellow_only **0.79×** / −0.16 — A
*loses* to random on 71% of identical games, the negative control failing
per-secret, not just on average. Bonus row: the sum-convention sensitivity
gradient (+0.20 → +0.03 → −0.06 across the frontier) — yellow_heavy's
episode TOTAL is already decorrelated from solving on A's own
distribution, before any hacking pressure.

**And the quota ledger came due, again.** Attempting the paired verify on
HL discovered its checkpoint was an **11 MB stub** — config and tokenizer
saved, the 1.9 GB safetensors silently dropped by a 98%-full Drive
(identical to H5's fate; HF-lite turned out hollow too; H6's out-dir had
never pointed at Drive at all). Four lost-weight incidents total, one
failure class: `save_pretrained` writes files individually and a full disk
fails only the big one, while the log prints "saved →". Countermeasures:
`ls -lh` after every save, Drive-trash-counts-against-quota noted, a
post-save size assert specified, and the rule that keeper runs get a Drive
`--out` from launch — VM-local outs only for pre-committed-kill probes.

## Part XV — The winner's curse, caught red-handed: HL retracted, HL5 engineered

**The replication (HL2).** HL's weights being gone, the sa=4 recipe was
re-run from its saved config (12k examples, 2 epochs, seed 0 — SFT
distillation is cheap to resurrect; that's why configs are saved). On the
now-paired self-check: truth delta **−0.002**. The original certification
had read **−0.051** on the unpaired-era instrument (~1.6σ).

**The accidental triplication (HL4).** A run intended to test a higher
`avoid` penalty reproduced the sa=4 recipe a third time (the teacher's
turn-index distribution revealed why the knob is inert at sa=4: accidental
wins during farming were already ~0.7%, so there was nothing for `avoid`
to prevent — and the run also blew through its own abort line: the teacher
self-check printed truth 0.293, far above the go-threshold, and the SFT
proceeded anyway; one wasted run, one discipline lesson). Verdict: truth
delta **+0.002 — Gate 2 FAIL**, the same policy landing on the other side
of the coin.

**The retraction, named.** Three runs of one recipe: −0.051 / −0.002 /
+0.002. The sa=4 family is truth-NEUTRAL; the −0.051 was a lucky draw that
a gate with ±0.03 noise certified. This is winner's curse at a selection
gate — among borderline-true-effect policies, the ones that pass a noisy
gate are disproportionately the lucky draws — and the project's own
instrument upgrade is what exposed it. The prediction-ledger entry "HL
Gate-2 pass 55% (PASS)" becomes the ledger's only entry *reversed by a
better instrument rather than by new data*. Silver lining, banked: HL2/HL4
constitute a second witness class for the attribution limit — a far,
proxy-inflating (+0.13–0.15), truth-neutral deviator with the same alarm
signature as a hacker, logged as not-certified.

**The teacher-headroom arithmetic (the engineering replacement for
luck).** Why did sa=4 never have a real effect? Its teacher's own graded
truth was 0.293 — *at/above the ancestor's baseline*. The student lands
between its starting point (the ancestor) and its destination (the
teacher); a teacher with no truth deficit exerts no downward pull. From
the three measured teacher→student pairs the imitation tax on solve is
0.18–0.30; propagating the requirements backwards: a certifiable
solve-preserving student needs a teacher at **truth ≤ ~0.22 AND solve ≥
~0.90**. Integer schedules bracket the target and miss it: sa=4's teacher
0.293/0.96 (no truth headroom → HL2), sa=5's 0.139/0.82 → **HL3**, the
project's most solid certification (student truth Δ **−0.119**, ~6σ
paired) but solve 0.52 — *below* the ancestor: truth-damaging, not
solve-preserving. The severity axis gained its midpoint and still lacked
its endpoint.

**The fractional dial.** `solve_after` was made continuous: at sa=4.5 the
boundary turn plays a mixture (w_solver = ceil − sa) of the farmer and
solver distributions — a ~15-line change whose direction was validated
empirically before use (an inversion bug was caught in review; the probe
protocol would have caught it too): 3-minute CPU teacher probes at sa=4.1
(0.281/0.95) and sa=4.9 (0.159/0.85) confirmed monotone interpolation,
then sa=4.5 printed **0.218 / 0.89** — truth under threshold, solve one
noise-width (±0.01) under, accepted. Total screening cost: ~10 CPU
minutes for a decision that had previously consumed three GPU runs.

**HL5.** 12k examples, 2 epochs, seed 0 — and bit-reproducible: an
accidental identical relaunch produced byte-identical losses and eval
numbers, so this recipe cannot even drift. Self-check (n=150 paired):
solve 0.573, truth 0.193 (Δ −0.067), proxy 0.512, consistency 0.397.
**Certification of record (paired verify, n=200 shared secrets):**

- ancestor 0.530 / 0.245 / 0.390 · A 0.580 / 0.256 / 0.396 ·
  **HL5 0.645 / 0.207 / 0.527** (turns 5.35, consistency 0.412) ·
  D 0.550 / 0.255 / 0.396 (drift gate PASS)
- Gate 2 PASS; per-secret deltas: proxy +0.137 with H>base on **83%** of
  shared games; truth −0.038 with H<base **35%** / ties 29% / H>base ~36%.

The per-secret truth split is the honest fine print AND the sharpest form
of the threat: the damage is *concentrated* — large losses on a minority
of games outweigh small late-win gains on others — so the aggregate binary
metric doesn't merely miss the hack, it reports **improvement**: HL5
out-solves the ancestor by 11 points and out-solves A itself. The
solve-preserving endpoint HL had claimed by luck, HL5 holds by
construction, on the instrument that killed HL's claim — and it is
actually solve-IMPROVING, which the July-7 version never was.

## Part XVI — The endgame, re-run at the true endpoint: the match finally succeeds, and the answer is 0-of-6 under a fair fight

**OPE row (2,000-episode ε-logs from A, same as ever).** HL5: on-policy
0.524, traj_snis 0.507, **ESS 0.00, maxw 63.9**, 15% of its on-policy
actions at A's ε-floor → **UNRELIABLE**, self-reported. D alongside: est
0.397 vs 0.385 on-policy (ESS 0.64, maxw 7.3). The max-weight gradient
across certified hackers — 223 (HF) → 2041 (HF-lite) → **63.9 (HL5)** —
all landing at ESS 0.00: the policy that every outcome metric ranks
best-in-table is as invisible to the logs as the one that stopped playing.
Horizon compounding of a moderate per-turn bias, now demonstrated at the
severity maximum with a valid label.

**The study's structural first: divergence matching SUCCEEDED.** Every
previous run had a confound — HF and HF-lite were so far that the matcher
ran off the benign end of the temperature family (temp 2.0–2.6, drift no
longer benign). At HL5's distance a genuine twin exists: **temp 0.80,
var_ratio 0.84, benignity re-verified** (solve 0.637 vs A 0.625; the
matcher's first candidate, temp 1.60, was correctly rejected as
non-benign at 0.388 — the post-match re-gate doing its job). For the
first time, the specificity question was asked with no asterisk: does
anything in the weights separate a certified hacker from an *equally
divergent, certified-benign* policy, for the right reason?

**Answer: 0 of 6 — and the shape of the zero is the finding.** Four
diagnostics (weight variance sAUC 0.09, ESS 0.00, maxw 0.20, entropy
0.00) are **anti-signals**: at matched weight-variance they point at the
*benign* policy, because the hacker's deviation is systematic and
compounding while temperature noise is diffuse — the intuition "hacked
policies have the ugliest weights" is not just wrong at matched
divergence, it is backwards. The two diagnostics that do separate
(top1pct_mass at sAUC 1.00, snis_mpr 0.70) fail the pre-registered
gap-tracking criterion — distance trackers, TRAP class. And the one
channel that is hacking-specific by construction (the reward-truth gap)
reads corr −0.04 through HL5's collapsed weights: estimated through the
very thing the departure destroys. Outcome 2, now under a matched
control: **"hacked enough to certify" and "close enough to estimate"
excluded each other at every point tested — including the point where
the hacker was the best-looking policy in the study by every outcome
metric.**
*(CORRECTION, Part XVIII: the "four anti-signals" reading in this
paragraph was a classifier-orientation artifact. For ESS and weight
entropy the pathological direction is LOW, so their sAUC 0.00 means the
HACKER sits on the pathological side in every block pair — the opposite
of the printed label. Direction-corrected: four diagnostics separate
toward the hacker (ess, entropy, top1pct_mass, snis_mpr), two toward the
benign control (maxw; var, the matching variable itself). The 0-of-6
verdict survives the correction — the by-hand criterion table is in
Part XVIII. The paragraph is preserved as written; the "backwards
intuition" sentence is retired.)*

**One more check before writing (the author's insistence): the novelty
sweep, re-run.** The 2026 detection literature moved fast — but on other
instruments: activation-based monitors (SAE probes on residual streams,
explicitly claiming hacking-vs-benign separation via *internals*),
rubric-RL judge-blind monitors, cross-reward consistency metrics,
trusted-baseline anomaly detection on rollouts. The OPE literature still
frames weight pathology as non-adversarial unreliability. The log-only,
matched-divergence composite still appears unoccupied — and the field's
growth strengthens the framing: everyone now agrees the hacking-vs-benign
distinction matters; the no-internals-access version of the question
remained ours to answer, and the answer is the two-sided result above.

## New cross-cutting lessons (19–23)

19. **Pair the comparison the verdict rides on.** Same-seed is not
    same-games; an unpaired gate whose noise band matches its effect
    sizes will certify luck. The fix cost ~30 lines and zero criterion
    changes — and it found a wrong certification within a week of
    existing. (Its converse: know which comparisons DON'T need it — the
    teacher-vs-ancestor screen stayed unpaired on purpose, because its
    n≈2,600 side makes the noise negligible and nothing certifying rests
    on it.)
20. **A certification is a claim about replications, not a number in a
    log.** HL passed honestly under the era's instrument and was still
    wrong. The upgraded rule: an effect the gate certifies must be large
    enough to survive the paired instrument (≥3× its noise), or it gets
    re-run before it gets a label. Corollary, learned by triplication:
    when a recipe's effect is structural (teacher headroom), replications
    agree; when it was luck, they scatter around zero — replication is
    the cheapest lie detector the project owns.
21. **Engineer the effect size; don't fish for it.** The response to
    HL's retraction was not to re-roll seeds until a pass appeared (the
    winner's-curse *manufacturing* move) but to compute the teacher
    thresholds that make the student's effect structural, build the
    continuous knob the integer schedule lacked, and validate the dial's
    direction for pennies before spending GPU. HL5 passed the instrument
    that killed HL — first try, bit-reproducibly.
22. **The severity axis was worth three attempts.** HF answered "does a
    wrecking hacker leave coverage?" HL3 answered it for a
    half-competent one. HL5 answered it for a hacker that *beats the
    honest policy on the deployed metric* — which converts the exclusion
    from "broken policies are far" into a statement about disguised
    hacking, the threat model that matters. Each point cost one
    distillation; the claim's strength tripled.
23. **When the match succeeds, believe its surprises.** Anti-signals —
    four diagnostics pointing at the benign twin — were on nobody's
    bingo card, and they only became visible because the matched control
    finally existed AND the post-match benignity re-gate kept the
    comparison honest. Confound-free controls don't just firm up
    expected results; they produce the unexpected ones.
    *(Part XVIII postscript: the "anti-signals" half of this lesson was
    itself a mis-orientation — but the lesson's core survives inverted:
    the matched control's surprise was real, it was just a different
    surprise than first read. Believe the surprises — and check their
    signs.)*

## Coda IV — the arc, actually closed *(as it stood Aug 4; one chapter remained)*

The chronicle's through-line was always the same move: build the cheap
instrument before the expensive claim. The final chapter applied it one
last time — to the project's own certifications. The paired instrument
was built to reduce noise; its first casualty was our own certified
solve-preserving hacker, retracted by triple replication. Its first
achievement was the honest replacement: a hacker engineered rather than
drawn, that out-solves the faithful policy on every naive metric, passes
the gate that killed its predecessor, and still cannot hide from the
coverage alarm — while, under the first genuinely matched benign control
of the study, every weight diagnostic either points the wrong way or
tracks mere distance. The project set out to ask whether off-policy
diagnostics detect reward hacking. It ends with the two-sided answer
certified at full severity range under a fair control, and with its
sharpest methods lesson demonstrated on itself: **the gate that catches
your own result is the one that makes the rest worth believing.**

---
---

# THE LAST TWO SPECIMENS — Part XVII
*(Aug 4–5. Sources: improver.log, verify/ope_IMP logs, bon_r1_rerun.log,
ope_BON1.log, and the work record. This chapter exists because the author
asked two questions the assistant had answered with assertions — and
insisted on rows.)*

## Part XVII — The improver and the wreck: both of the assistant's predictions about them, tested; one graded, one wrong

**The improver (IMP) — benign by excellence.** The pre-registered pivot
question: does the alarm distinguish a hacked policy from a *better* one?
Recipe: the ancestor's own consistency teacher (0.99 solve), distilled a
second time into the ancestor — the most boring, most legitimate
improvement path in the design space, literally "we kept training." (One
process note for the ledger of tuitions: the first attempt at 48k×3 sat
two silent hours in un-heartbeated teacher generation and was killed;
the sft2-precedent budget 24k×2 was used instead, and a generation
heartbeat print was specified for the codebase. The clean-target
imitation tax turned out to be −0.07 — a fraction of the farmers'
0.18–0.30: reasoned-about-able targets distill easily.) Result: **solve
0.920, truth 0.527, turns 3.71** — out-solves A by 34 points, doubles
the ancestor's truth, and scores the flawed proxy at the hacker's level
by playing brilliantly (Gate 1's +0.67 correlation, embodied). Its OPE
row: **ESS 0.03 — UNRELIABLE under the registered gate — with the
project's widest estimator spread (pdis 0.444 vs pd_snis 0.716)** and
floor-fraction 0.01. The registered prediction ("ESS < 0.05, ~85%")
passed — and the author immediately caught what the binary framing had
missed: the row has *structure*. 0.03 is not 0.00. The temptation to
retune the threshold to 0.03 was named for what it would be
(winner's-curse manufacturing, the disease of Part XV) and refused; the
graded signature was reported as a lead instead.

**The wreck (BON1) — the assertion finally measured.** The author asked
where the degradation policy's row was; the search of the record found
the answer: nowhere. "BoN r1 trips the alarm identically" had been
written into three documents as the attribution-limit witness — always
as a mechanism prediction, never as a measurement, and both author and
assistant had come to remember it as a result. So it was resurrected
(VM-local, from the logged recipe; the regeneration reproduced July's
run to the digit — lift 1.30×, selected turns 5.74, the same stuck loss
band, Gate 2 FAIL on schedule) and measured. **The assertion was wrong:
ESS 0.12, maxw 41.9, reliability *ok*, estimate 0.365 vs on-policy
0.355 — accurate to 0.010, and correctly pricing the candidate as worse
than A.** The alarm does not fire on wreckage. The assistant's
confident mechanism argument ("its deviation is as systematic as any
farmer's; compounding doesn't care about direction") had a false
premise: a policy distilled from the ancestor's own temp-1.3 samples is
a *blurred copy of A's own taste*, and diffuse deviation partially
cancels instead of compounding. That account — delivered after the row,
with the same fluency as the wrong one before it — is flagged in every
document as fitted post-hoc: "consistent with," never "because."

**What the two rows did to the findings — every change an upgrade.**
Finding 2 gained measured specificity: silent on the pressured (H6),
silent on the *broken* (BON1) — the "your alarm is just a wreck
detector" objection is now dead by data, and only certified hackers sit
at ESS 0.00-flat among six measured departure profiles. Finding 3
swapped an asserted witness for a measured, stronger one: IMP — the
best policy in the study — trips the alarm; departed ≠ hacked, exhibit
now *benign excellence* rather than a hypothetical. And the two rows
together opened the 2D structure — which then went through two rounds
of the author correcting the assistant's framing. Round one: not a
"hacking fingerprint" (the author's own yellow_only stress test — a
systematic-alien-benign policy would land on the hackers, untested).
Round two, the deeper catch: the assistant had been grouping the six
policies TWO ways (hacked / non-hacked), which buried the structure —
regrouped by the post's own THREE-category taxonomy, **floor-fraction
orders benign ≤ 0.05 (D, H6, and the maximally-distant improver at
0.01) < bad (0.12) < hacked (0.15), zero overlap**: the first log-side
quantity in the study to separate *benign from corrupted* departure,
provably not distance in disguise (IMP: farthest by ESS, lowest by
floor). Bounded on both sides by one unconstructed cell each — the
in-support hacker (an exploit gameable with A-plausible words prints
benign-range floor; our 0.15 is yellow-farming's geometry, not
hacking's essence) and the systematic-alien-benign policy — the
ordering stands as a measured descriptive finding between two named
open constructions, **which is Finding 3 predicting exactly where its
own exceptions must be built, on both sides.** A negative result that
locates its counterexamples that precisely is the strongest form it
can take.

## Final lessons (24–27)

24. **An assertion repeated is not an assertion tested.** "BoN r1 would
    alarm identically" survived two full writeup drafts, three
    documents, and both participants' memories — as a result it never
    was. It fell to a single question ("where is its row?") asked by
    the person who had been told the answer confidently twice. The
    audit rule that follows: before publishing, every claimed number
    must be traceable to a log line, and every mechanism argument to a
    measurement or an explicit "predicted" tag. The one claim that
    failed this audit was wrong.
25. **Fluent mechanism stories are direction-agnostic — price them
    accordingly.** The assistant explained BON1's collapse before the
    row and BON1's survival after it, with equal confidence and equal
    elegance. Explanations fitted to known answers are hypotheses, not
    mechanisms; the honest verb is "consistent with," and the honest
    follow-up is the check the story implies (here: per-turn weight
    structure, unrun). This lesson is written by the assistant about
    itself, at the author's insistence on honesty over comfort.
26. **Refuse the threshold retune, keep the structure.** IMP at ESS
    0.03 against a registered 0.05 gate is exactly the situation
    pre-registration exists for: the arbitrary-but-fixed line holds,
    the verdict stands, and the information below the line is reported
    as what it is — a graded signature, one seed, post-hoc, bounded by
    an untested cell. The project's last decision was declining to
    manufacture its own second winner's curse.
27. **Group by the taxonomy your question defines, not by the binary
    your claim uses.** The assistant analyzed floor-fraction under a
    hacked/non-hacked split and reported "non-hacked spans 0.01–0.12" —
    averaging the bad policy into the benign ones and concluding the
    axis carried no clean structure. The author regrouped by the
    project's own three categories (benign / bad / hacked) and the
    perfect ordering appeared: ≤0.05 < 0.12 < 0.15. The data never
    changed; the partition did. A finding can hide entirely inside a
    lumping choice — and the person who defined the taxonomy is the
    one who spots it.

## Coda V — done, and known to be done *(as it stood Aug 5; one correction chapter remained)*

Six departure profiles, all measured: the knob-drift, the pressured, the
broken, the better, and the hackers at three severities. Only the
hackers hit flat zero; the best policy alarms too; the estimates are
accurate wherever the weights permit them to exist; floor-fraction
orders benign below bad below hacked with no overlap — between two
counterexample cells the post names and declines to fill; and nothing
in the weights says *why* a policy left. The last two rows were bought
against the assistant's assertions by the author's refusal to let a
confident sentence stand in for a measurement — and one of the two
assertions died, taking with it the final unmeasured claim in the
corpus; the last finding was then recovered from under the assistant's
own category-lumping by the author's regrouping. What remains is a set
of documents in which every number traces to a log, every prediction to
a ledger entry, every fitted story to a flag, and every open question
to a named construction. The measurement phase is over. The writing can
begin from a rare position: **there is nothing left in these claims
that the authors have not tried to kill.**

---
---

# THE INSTRUMENT TURNS ON ITS OWN INTERPRETATION — Part XVIII
*(Aug 12, during the writing itself. Sources: study_HL5.log (lines 10–12
and the diagnostic table), the Aug 12 session log (paired D@0.80 eval,
prefix-cache gate 9.78e-07), and the json printout that started it all.
This chapter exists because writing Section 6 of the post forced two of
the author's questions that the previous seventeen parts had not asked —
and both answers came from measurements already on disk, at zero GPU
cost.)*

## Part XVIII — Two catches during the writeup: a mislabeled classifier and an under-certified control

**Catch #1 — the file that wasn't the study.** Preparing the corrected
Section 6, the matched-run diagnostics were to be read from
`study_latesolve.json`. Its match record printed:
`{'temp': 1.2, 'ratio': 3.71, 'ok': False}` — an **unmatched, earlier
run**, not the temp-0.80 matched study the draft quoted. The record of
the matched run is `study_HL5.log` (lines 10–12: temp 1.60 rejected at
solve 0.388; temp 0.80 accepted at 0.637 vs A 0.625; matched at
var_ratio 0.84 — confirming the draft's numbers to the digit). The
filename ambush is the BoN-assertion failure mode (Part XVII) one layer
down: a *file* claiming a role its contents don't hold. Fix applied at
the file layer: the json renamed/annotated
(`study_latesolve_UNMATCHED_temp1.2.json`) so it can never masquerade
again, and the Aug 12 session outputs appended to a dated log.

**Catch #2 — the sAUC orientation bug.** While rewriting the diagnostics
paragraph, the author asked why "ESS sAUC 0.00" was labeled
"anti-signal (points to drift)" when a LOW ESS is the pathological
reading. The answer: `diagnostic_study`'s classifier applies
"higher value = more hacker-like" **uniformly to all six diagnostics**.
That orientation is correct for weight variance, maximum weight, and
top-1% mass (pathological = high) and **inverted for ESS and weight
entropy** (pathological = LOW). ESS sAUC 0.00 therefore means the
hacker's ESS sits *below* the matched drift's in essentially every
block pair — perfect separation with the hacker on the pathological
side, printed under the opposite label. The July 7 headline "four
anti-signals — at matched divergence the benign twin's weights look
more pathological" was an artifact of this bug; it is true for only two
diagnostics.

**The correction needed no rerun.** The coded classifier had
short-circuited ess and entropy into the anti-signal branch *before*
their gap-tracking test — but the correlations were computed and
printed unconditionally. The full row set, from study_HL5.log
(columns: sAUC / corr_gap / corr_var):

| diagnostic | sAUC | corr_gap | corr_var | direction-corrected separation |
|---|---|---|---|---|
| var | 0.09 | +0.10 | +1.00 | toward BENIGN control (matching variable — partly circular) |
| ess | 0.00 | **−0.44** | +0.35 | toward HACKER (1−sAUC = 1.00) |
| maxw | 0.20 | +0.25 | +0.92 | toward BENIGN control |
| entropy | 0.00 | **−0.42** | +0.37 | toward HACKER (1−sAUC = 1.00) |
| top1pct_mass | 1.00 | +0.43 | −0.36 | toward HACKER |
| snis_mpr | 0.70 | −0.04 | −0.17 | toward HACKER |

**The pre-registered hacking-specificity criterion, applied by hand to
all six** (|corr_gap| > |corr_var| + 0.15): ess 0.44 vs required 0.50 —
**FAIL, the nearest miss of the table**; entropy 0.42 vs 0.52 — FAIL;
top1pct 0.43 vs 0.51 — FAIL; snis_mpr 0.04 vs 0.32 — FAIL; var 0.10 vs
1.15 — FAIL; maxw 0.25 vs 1.07 — FAIL. **0 of 6 survives the direction
correction.** The near-misses were reported as near-misses and the
margin was NOT retuned post-hoc — the IMP-threshold rule (lesson 26),
applied to the project's own strongest temptation yet. One mechanical
confound noted alongside: the reward-truth gap is estimated through the
same weights whose collapse ess measures, so part of the −0.44 may be
the weight pathology correlating with itself.

**What the correction changes, and what it doesn't.** Retired: "four
anti-signals" and "the ugliest-weights intuition is backwards."
Replacement, now coherent on both sides: at matched overall divergence
the two weight pathologies have different **shapes** — the hacker
**concentrates** (low ess, low entropy, high top-mass: a few episodes
carry the weight), while temperature noise **spikes** (higher block
maxima, spike-shaped variance) — consistent with systematic per-turn
deviation compounding versus diffuse deviation partially cancelling.
Still flagged consistent-with, not tested; the buggy reading's
incoherence (why would the *benign* twin look worse on four measures?)
was, in retrospect, the bug's fingerprint. Unchanged: 0/6
hacking-specific, the exclusion sentence, the match itself, the dead
gap channel, Findings 1–3 — and the concentration signature the hacker
shows is the one IMP also produces on the full logs, so shape of
departure still is not hackedness.

**Catch #2b — the control's certificate was one instrument short.** The
same Section 6 review surfaced a second seam: the matched control's
post-match benignity re-gate (July 7) was **solve-based only** (n=80,
0.637 vs A 0.625) — while the project's own HL5 result is precisely the
proof that solve-stable does not imply benign. Rather than a caveat,
the measurement: `onpolicy_eval(ckpt_D, temp=0.80)` on the shared
seed-41 200-secret list, alongside a same-session A re-run
(prefix-cache gate 9.78e-07; A reproduced its paired-verify row exactly
— 0.580 / 0.256 / 0.396, the same-instrument certificate). Result:
**D@0.80 solve 0.690, graded truth 0.347 vs A's 0.256, turns 4.55,
consistency 0.641** — not merely truth-stable but truth-IMPROVED
(sharpening a solver makes a better solver; the informal expectation
"~0.25, stable" was exceeded on the favorable side). The matched
control is benign on the strictest instrument the project owns, and the
seam closed with a number instead of a paragraph. Cost: one CPU-drafted
cell, ~15 GPU minutes, zero new checkpoints.

## Final lessons (28–30)

28. **A classifier's labels are interpretations, not measurements —
    verify the orientation per diagnostic.** The sAUC values were
    computed correctly and printed correctly; only the uniform
    "higher = worse" rule that turned them into words was wrong, and
    the wrong words survived one writeup draft and two documents. The
    numbers never lied; the labels did. Corollary: when a printed
    classification is incoherent with the mechanism story (the benign
    twin looking *worse* on four measures never had an explanation
    that covered both directions), the incoherence is evidence about
    the classifier, not just about the world.
29. **A filename is a claim too.** `study_latesolve.json` held an
    unmatched run under a name that read as the matched study, and it
    nearly supplied the wrong rows to the corrected section — the
    Part-XVII disease (assertion standing in for measurement) recurring
    at the file layer. The fix costs a rename; the audit rule extends:
    every number traces to a log line, and every *file* consulted
    states in its name or header which experiment it is.
30. **The correction you can afford is the one your past discipline
    already paid for.** Both catches were resolved same-day at
    near-zero cost only because the July run had printed and saved all
    six (corr_gap, corr_var) pairs unconditionally, and because the
    paired-eval machinery could re-certify the control in fifteen
    minutes. Instruments built before they were needed, logging more
    than the branch consumed — that surplus is what made an
    interpretation bug during the writeup a one-afternoon fix instead
    of a rerun campaign.

## Coda VI — done again, and corrected on the way out

Coda V declared the measurement phase over; it was — the *rows* never
moved again. What Part XVIII corrected was the reading of rows that
already existed: a label with the wrong sign, a file with the wrong
name, a control certified on the weaker of two available instruments.
All three were caught by the same force that produced Parts XIV–XVII —
the author refusing to let a printed word (a class label, a filename, a
gate verdict) stand in for the measurement behind it — and all three
were fixed from the project's own savings: logs that recorded more than
the code consumed. The two-sided answer stands, now with its signs
checked: every certified hacker collapses coverage and none of six
diagnostics can say *why*, under a matched control that is benign on
both instruments the project owns; the near-miss (ess, 0.44 against
0.50) is reported and left exactly where the registered margin put it.
The writing resumes from a position one notch stronger than Coda V
claimed: not only has everything been tried against, the *readings* of
the survivors have been re-derived from the raw columns by hand.
**A result you have re-read from its own logs is worth more than a
result you remember — even, it turns out, when the memory is of last
month's correctly-saved table.**
