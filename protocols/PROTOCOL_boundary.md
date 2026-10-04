# Experiment 1 — the grid boundary at 2·10⁻² (TMLR)

Written, committed and pushed before any run of this campaign. Reuses no earlier run as a result: the only earlier runs it reads are the
selection runs that the paper's own selection used (seeds 0 and 1 at the other rates of the main grid), because applying the selection rule
unchanged requires them; they are read, not rerun, and this is declared here.

**Question.** The paper's main grid stops at 10⁻². For four cells, the rate selected at 10⁻² was never checked at 2·10⁻²; the paper says they
« remain unchecked ». Does the selected rate stay the same once 2·10⁻² is run?

**Cells.** Llama-3.2-1B, format task: `top`, `bottom` and `random` at rank 1, and `random` at rank 4.
**Configuration.** That of each cell's reported runs, copied from its reported run at seed 2 (same data, 600,000 supervised tokens,
evaluation, checker, optimiser, clipping, modules, and for `random` the same subspace draw rule), except the rate and the seed.
**New runs.** Each cell at 2·10⁻², seeds 0 and 1 (8 runs). Pinned to the RTX 4000 Ada nodes, as the main campaigns.

**Selection rule (the paper's, unchanged).** The selected rate maximises the mean validation score over seeds 0 and 1, over the main grid of
the cell plus 2·10⁻². **Tie rule** (the paper's addendum rule, made explicit): if the two best rates differ by 0.005 or less in that mean,
both are run on three added selection seeds, 30 to 32, and the rate that maximises the mean over seeds 0, 1 and 30 to 32 is retained if it
leads by more than 0.005; if it does not, the rate with the best five-seed mean is retained and declared an unresolved tie.

**If a cell's selected rate changes**, it is declared, the cell is rerun at the new rate on seeds 2 to 6 (5 runs), and both the old and the
new reported values are given. If it does not change, no reported run is made.

**Reported.** For each cell: the validation means at every rate of the grid and at 2·10⁻²; the selected rate before and after; whether a
tie arose and how it was settled; and, if the rate changed, the old and new scores on seeds 2 to 6. No test is pre-registered: the outcome
is the selection itself. Compute: about 1 PARTITION-hour, more if a tie or a change occurs.
