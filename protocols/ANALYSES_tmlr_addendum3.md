# Third addendum to ANALYSES_tmlr.md — A3c, committed before it is executed

Written after A3b was run (results committed in `<commit>`). A3 and A3b are kept unchanged.

**What A3b left open.** 38 published values were reproduced by two or three cells. A diagnostic (03/10) showed that none are exact
reruns: the candidate cells differ in arm, rank, task, model or seed set, and their scores differ seed by seed. They are coincidences of
the 4-decimal mean (on OpenBookQA, a 5-seed mean moves in steps of 0.0004). The cell behind a published value is the one named by the
table's row and column, which A3b does not read.

**A3c.** For each occurrence of an ambiguous value (54 occurrences, value × table), the table, row and column were read in
`analyses_inputs/main_tmlr.tex` and written, before execution, into `analyses_inputs/a3c_map.json`: task, model, arm, rank, and the seed
set where the table states it. A value may name two legitimate cells in one table (0.5924: `top_unfrozen` and `bottom_unfrozen` at rank 1
on Mistral-7B; 0.7450: `top` and `bottom` at rank 8 on the format task) or different cells in different tables (0.5524, 0.5528). A3c keeps,
among A3b's candidates, those that match; any occurrence that does not resolve to exactly the expected cells is reported as a problem.
Unambiguous values keep their A3b cell. Each distinct cell (a distinct run set) is counted once, whatever the number of tables it appears
in, and the PARTITION summary is computed on these cells only. Outputs: `analyses_tmlr_a3c.md` and `analyses_tmlr_a3c.json`. Status: post hoc,
descriptive.

**Not covered.** 0.5905 is a value quoted in the captions of `tab:gradacc` and `tab:warm` (a format-task reference retrained on one PARTITION
model); A3b found no run set for it. It is reported as not found.
