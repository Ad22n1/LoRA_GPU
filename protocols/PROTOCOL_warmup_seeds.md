# Experiment 5 — the warm-up on two more direction seeds (TMLR)

Written, committed and pushed before any run. Reuses no earlier run as a result; the test compares with the published `top` and uses the
G fixed in advance, as constants, because it repeats the test of `PROTOCOL_warm.md` unchanged (declared here).

**Question.** The warm-up campaign froze B on the output directions of a free LoRA after 5, 10 and 25 % of its training, with one direction
seed (0). Does the dose-response hold for other direction seeds?

**Direction runs.** On both pairs (OpenBookQA with Qwen2.5-1.5B; the format task with Llama-3.2-1B), a free LoRA at rank 2 at 2·10⁻³, as in
`PROTOCOL_warm.md`, with two new direction seeds, **301 and 302**, never used for a direction run (direction runs so far used seeds 0 and
7). Factors saved at 5, 10 and 25 % of training (steps 3, 5 and 13 of 49 on OpenBookQA; 14, 28 and 69 of 274 on the format task); bases
built from them exactly as in the warm-up campaign, and their MD5 committed before the arms run.
**Arms.** For each new direction seed and pair: B frozen on `warm5`, `warm10` and `warm25`, A trained for the remaining steps, each at the
rate the original warm-up campaign selected for that placement and pair (read from its selection files and written into the check file
before the first run), seeds 2 to 6. 4 direction runs and 60 arm runs, on RTX 4000 Ada.

**Pre-registered test.** The test of `PROTOCOL_warm.md` on `warm10`, for each new direction seed and pair (4 tests, **Bonferroni over these
4**): warm10 − `top` (published: 0.4652 on OpenBookQA, 0.6443 on the format task), paired over seeds 2 to 6, two-sided t > 4.315 (4 d.f.,
level 0.05/4); P = (warm10 − `top`) / G, G fixed in advance (8.04 and 12.41), read on its interval at level 1 − 0.05/4: **suffices** if the
lower bound reaches 0.7, **does not suffice** if the upper bound stays below 0.3, **inconclusive** otherwise.
**Reported.** P for `warm5`, `warm10` and `warm25` for every direction seed, including the original seed 0 (from the warm-up campaign), so
that the dose-response has three direction seeds per placement. Compute: about 15 to 20 PARTITION-hours; the first experiment to drop if compute
is short.
