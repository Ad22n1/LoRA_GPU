# A frozen on a random orthonormal basis (dual_random_ortho)

Written, committed and pushed before any run of this campaign. It runs alongside the TMLR review and modifies neither the paper, nor the
supplementary record, nor any existing result. Reuses no earlier run as a result of this experiment.

**Question.** The paper freezes A only on the top and bottom spectral bands (`dual_top`, `dual_bottom`). Zhu et al. (2024) freeze A on
random factors. Does the advantage of freezing A over freezing B hold when A carries no spectral information at all?

## The arm

`dual_random_ortho`, as implemented in `src/lorasub/lora.py` (mode list `DUAL`). In every adapted module, A (rank r by d_in) is frozen on a
random orthonormal basis of the input space: the Q factor of a Gaussian matrix of shape d_in × r, transposed. It is drawn exactly as the
existing `random_ortho` arm draws B: same generator, seed `(s · 7919 + 17 + h) mod (2⁶³ − 1)`, where s is the run's seed (no separate
`subspace_seed` is set, as in the published `random_ortho` runs) and h a hash of the module's name (`subspace_salted = True`), so each module
gets its own basis and each run seed its own draw. B starts at zero and is trained. No SVD is used.

**Earlier attempts, declared.** This arm was launched once before, under `PROTOCOL_freeze_A.md`: its 49 runs all failed before training on
a misplaced check (found and corrected on 26/09; comment in `lora.py`). Completed `dual_random_ortho` runs also exist under `~/lora-runs`
(seen by analysis A3b). None is used here: every run of this experiment is new, under the new root `~/lora-runs-randA`.

## Trainable parameters, fixed before any run

A dual arm trains B, r × d_out per module: the same count as `dual_top` and `dual_bottom` at the same rank.

| module | d_out (Llama-3.2-1B) | r = 2 | d_out (Qwen2.5-1.5B) | r = 2 | r = 16 |
|---|---|---|---|---|---|
| q_proj | 2,048 | 4,096 | 1,536 | 3,072 | 24,576 |
| k_proj | 512 | 1,024 | 256 | 512 | 4,096 |
| v_proj | 512 | 1,024 | 256 | 512 | 4,096 |
| o_proj | 2,048 | 4,096 | 1,536 | 3,072 | 24,576 |
| gate_proj | 8,192 | 16,384 | 8,960 | 17,920 | 143,360 |
| up_proj | 8,192 | 16,384 | 8,960 | 17,920 | 143,360 |
| down_proj | 2,048 | 4,096 | 1,536 | 3,072 | 24,576 |
| per layer | | 47,104 | | 46,080 | 368,640 |
| **total** | 16 layers | **753,664** | 28 layers | **1,290,240** | **10,321,920** |

The campaign reads each run's own count (`n_trainable`) from its selection runs and **stops before any reported run** if one differs.

## Configuration

Everything not stated here is as in the main campaigns: data, 600,000 supervised tokens, evaluation, checker, optimiser, gradient clipping,
the seven adapted module types, α = r. Each cell copies the configuration of a published reference run of its pair (the published
`random_ortho` run at rank 2; the published `dual_top` run at rank 16), changing only the mode, the rank, the rate, the seed, the salting
(`subspace_salted = True`) and the output root. All runs on RTX 4000 Ada (`scripts/nodes_for_gpu.py`), under `~/lora-runs-randA`.

**Selection.** The paper's rule: on seeds 0 and 1, the rate maximising the mean validation score over the main grid with its upper
extension (5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻²); a selected rate must be an interior maximum, so if the best rate is at an edge the
grid is extended by one step on the 1-2-5 ladder. **Tie rule:** if the two best rates are within 0.005, both are run on seeds 30 to 32; the
rate with the best mean over seeds 0, 1 and 30 to 32 is retained if it leads by more than 0.005, otherwise the best five-seed mean, declared.
**Reported** on seeds 2 to 6.

## Part 1 (primary): rank 2, both main pairs

The format task with Llama-3.2-1B; OpenBookQA with Qwen2.5-1.5B.

**Pre-registered test (P1).** `dual_random_ortho` − `random_ortho` > 0, where `random_ortho` is B frozen on a random orthonormal basis,
the published arm on the same seeds 2 to 6 (format 0.6534; OpenBookQA 0.4592). One test per pair, paired over seeds 2 to 6, two-sided,
Bonferroni over the 2 pairs: **t > 3.495 (4 d.f.)**. Comparing with a published arm, whose runs are read and not rerun, is declared here.

**Descriptive**, with 95 % paired intervals over seeds 2 to 6: `dual_random_ortho` − `dual_top`, − `dual_bottom`, and − the
comparable-budget free LoRA at rank 1 (published arms: format 0.7739, 0.7818, 0.7684; OpenBookQA 0.5308, 0.5248, 0.5456).

**Reading fixed in advance.** If P1 holds on both pairs, the advantage of freezing A over freezing B does not depend on a spectral
placement of A. If it holds on one pair only, the report says so, pair by pair. If it holds on neither, the advantage of freezing A depends
at least in part on its spectral placement.

## Part 2 (secondary, descriptive only): rank 16, OpenBookQA with Qwen2.5-1.5B

`dual_random_ortho` at rank 16, rate selected as in Part 1, seeds 2 to 6. Reported next to the published `dual_top` (0.5544) and
`dual_bottom` (0.5364) at rank 16 and the comparable-budget free LoRA at rank 7 (0.5616), with 95 % paired intervals. No test.

## Published arms

Each published arm is identified by its published score on seeds 2 to 6 (the mean of one run per seed reproducing it within 10⁻⁴); its
per-seed values are the ones paired. If one cannot be identified, the campaign stops before any run.

## Report

The protocol push time; the trainable counts; the selected rates with any tie or extension; the scores per seed and their mean; the P1 test
with its status; the descriptive comparisons; the compute used (about 6 to 7 PARTITION-hours expected); any deviation, declared as such.
