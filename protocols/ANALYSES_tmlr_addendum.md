# Addendum to ANALYSES_tmlr.md — precisions and one deviation, committed before A2 and A3 are executed

Written after A1 was run (results committed in `<commit>`) and before A2 and A3 are run. A1 is unchanged.

1. **A2, source of σ_d (deviation, declared).** For every comparison but two, the paper prints both the mean difference and t (or a
   bootstrap interval); σ_d is taken from those printed values, with the place they are printed (`analyses_inputs/a2_spec.json`, field
   `source`). For rows 10 (β = 0.75 against the four other positions, first five seeds of Qwen2.5-0.5B) and 110 (Mistral-7B, band gap
   with B trained against B frozen), the paper prints t only. ANALYSES_tmlr.md said σ_d would then be computed from the runs; it is not:
   identifying those cells in the runs needs the position parameter of the sweep and the trained-B arms of the Mistral campaign, which the
   value-matching of A3 does not provide reliably. For these two rows A2 reports the standardised effect d_z only, and says so.
2. **A2, intervals.** Rows 35 and 125 print a bootstrap interval, not a mean and a t: the MDE is (z_α + z_0.80)·SE with SE = half-width / 1.96,
   a normal approximation, stated in the output.
3. **A3, identification of the cells.** Completed runs are grouped by identical configuration, every field except the seed, the output
   path, the notes, the saved factor steps, the SVD-cache path and the data fingerprint. For each group and each reported seed set of the
   paper (2-6, 2-8, 9-16, 7-16, 2-16, 120-124, 22-26, 42-46, 140-144, 50-64, 65-104), the mean of the task score is compared with every
   4-decimal number in the tables of the main paper (`analyses_inputs/main_tmlr.tex`); a cell is found when they agree within 5·10⁻⁵.
   When a seed has several runs in a group, the earliest is tried first, then the latest. Numbers that match no cell are listed; not all of
   them are cell scores.
