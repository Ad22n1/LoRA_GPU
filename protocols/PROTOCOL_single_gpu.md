# Experiment 2 — rank 2 on a single PARTITION model (TMLR)

Written, committed and pushed before any run. Reuses no earlier run: every arm is rerun.

**Question.** Do the paper's rank-2 results on the format task hold when every run of the comparison runs on one PARTITION model?

**Population.** Llama-3.2-1B, format task, rank 2 (`top`, `bottom`, `dual_top`, `dual_bottom`), the budget-matched free LoRA at rank 1,
and B trained at rank 1 on both bands (`top_unfrozen`, `bottom_unfrozen`). Each arm at its **reported rate**: the rate of its reported cell,
found by its published score on seeds 2 to 6 (`top` 0.6443, `bottom` 0.5922, `dual_top` 0.7739, `dual_bottom` 0.7818, free LoRA 0.7684,
`top_unfrozen` 0.7928, `bottom_unfrozen` 0.7954), and written by the campaign into its check file, committed before the first run.
Configuration copied from each cell's reported run at seed 2. Seeds 2 to 6 (35 runs). **All on one PARTITION model, RTX 4000 Ada**, recorded
by every run; a run on any other model is discarded and rerun, and the count is reported.

**Tests (the paper's, repeated).** Paired over seeds 2 to 6, per band:
1. G = free LoRA − B frozen > 0, two-sided t, Bonferroni over the two bands: t > 3.495 (4 d.f.).
2. A frozen − B frozen > 0, same threshold.
3. R = B trained (rank 1) − B frozen, read against 0.7 G as in the paper: established if the whole 95 % interval lies above 0.7 G.
**Reported.** Each value next to the published one, and whether every result established in the paper stays established. Compute: about
4 PARTITION-hours.
