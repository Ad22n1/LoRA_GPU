# Experiment 4 — separating the frozen factor from the module shape (TMLR)

Written, committed and pushed before any run. Reuses no earlier run.

**Question.** Arms that freeze A put most of their update in `gate_proj` and `up_proj`; arms that freeze B put more of theirs in
`down_proj` (analysis A2 of 29/09). The cost of freezing B could therefore come from the shape of the modules rather than from the frozen
factor. We test the factor **inside each module type**.

**Pairs.** The format task with Llama-3.2-1B; OpenBookQA with Qwen2.5-1.5B. Everything not stated here is as in the main campaigns (data,
600,000 supervised tokens, evaluation, checker, optimiser, clipping at 1.0, α = r), copied from each pair's reported runs.
**Groups, each run separately, adapting nothing else:** (G1) `down_proj` only, in every layer; (G2) `gate_proj` and `up_proj` only, in every
layer.
**Arms.** G1, rank 2: `top`, `bottom` (B frozen on the dominant and minor bands), `dual_top`, `dual_bottom` (A frozen on the same bands).
G2: the same four at rank 2, plus `top` and `bottom` at equal trainable parameters with the A-frozen arms: rank 8 on Llama-3.2-1B (16,384
against 16,384 per module) and rank 12 on Qwen2.5-1.5B (18,432 against 17,920, +2.9 %). In both groups, a free LoRA at rank 1, reported
descriptively only (budget-matched in neither group).

**Trainable parameters, fixed before any run** (per module / total; Llama-3.2-1B: 16 layers, hidden 2048, intermediate 8192;
Qwen2.5-1.5B: 28 layers, hidden 1536, intermediate 8960):

| group | arm | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|
| G1 | B frozen, r = 2 | 16,384 / 262,144 | 17,920 / 501,760 |
| G1 | A frozen, r = 2 | 4,096 / 65,536 | 3,072 / 86,016 |
| G1 | free LoRA, r = 1 | 10,240 / 163,840 | 10,496 / 293,888 |
| G2 | B frozen, r = 2 | 4,096 / 131,072 | 3,072 / 172,032 |
| G2 | A frozen, r = 2 | 16,384 / 524,288 | 17,920 / 1,003,520 |
| G2 | B frozen at equal parameters | r = 8: 16,384 / 524,288 | r = 12: 18,432 / 1,032,192 |
| G2 | free LoRA, r = 1 | 10,240 / 327,680 | 10,496 / 587,776 |

The campaign recomputes these counts from each run's model before launching, and stops if one differs.

**Rates.** Selected per arm and per group on seeds 0 and 1, on the main grid with its upper extension: 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³,
10⁻² and 2·10⁻². A selected rate must be an interior maximum: if the best rate is at an edge, the grid is extended by one step on the 1-2-5
ladder, as in the paper. Tie rule of experiment 1 (two best rates within 0.005: seeds 30 to 32, a lead above 0.005 needed, else the best
five-seed mean, declared). **Reported on seeds 2 to 9** (eight seeds), at the selected rates. All runs on RTX 4000 Ada.
**Floor.** The base model is evaluated once per pair (no training). If, in a group and pair, the free LoRA gains less than 2 points over
the base model (mean over seeds 2 to 9), that group and pair are declared not testable.

**Pre-registered predictions.** (P1) In G1, at rank 2: `dual_top` − `top` > 0 and `dual_bottom` − `bottom` > 0. (P2) In G2: `dual_top` −
`top` > 0 and `dual_bottom` − `bottom` > 0, against the B-frozen arms at equal parameters (rank 8 or 12). **Tests:** paired over seeds 2 to 9
(7 d.f.), one per pair, band and group: **8 tests, Bonferroni over these 8**. For each, the two-sided interval at level 1 − 0.05/8
(t = 3.855 at 7 d.f.): **established** if it lies above zero; **reversed** if it lies below zero; **inconclusive** otherwise. The rank-2
comparison in G2 (`dual_*` against `top`/`bottom` at rank 2) is reported descriptively.
**Reading fixed in advance.** If P1 and P2 hold on both pairs, the cost of freezing B comes neither from the module shape nor from the
parameter count. If they hold in one group only, the paper must say that the module shape explains part of the effect. Any other pattern
is reported as it is, without a reading.
Compute: about 55 to 60 PARTITION-hours.
