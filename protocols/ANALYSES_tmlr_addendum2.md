# Second addendum to ANALYSES_tmlr.md — A3b, committed before it is executed

Written after A3 was run (results committed in `<commit>`). A3's outputs are kept unchanged; A3b is added beside them.

**Defect found in A3.** A3 grouped runs by their complete configuration, field by field. A cell whose seeds differ only by fields that do
not enter a run's identity (`eval_limit`, `eval_forgetting`, notes, paths) or by fields added to the code later (absent in older runs)
was split across groups, none of which had all its seeds. Three published values matched no cell, two of them central (0.6443 and
0.5922, `top` and `bottom` at rank 2 on the format task), and 33 values matched two or three cells that could be fragments of one cell
or true coincidences.

**A3b.** Same question, same inputs, three changes:
1. runs are grouped as the trainer identifies them: the fields of `lorasub.config._NON_ID_FIELDS` are ignored; fields added later take,
   when absent, the value they had before (`max_new_tokens` 128, `data_fingerprint` empty, `token_weighted_loss` false);
   `subspace_salted` is compared for the salted modes only (`lorasub.lora.SALTED_MODES`), absent meaning false;
2. runs of the TMLR campaigns (`~/lora-runs-tmlr`) are excluded: A3b concerns the runs behind the paper;
3. when a seed has several runs in a group, every one-run-per-seed combination is tried; every cell and every run set reproducing a
   published value is reported, and a value reproduced by several cells is listed with all of them and their PARTITION compositions, rather
   than settled.
Outputs: `analyses_tmlr_a3b.md` and `analyses_tmlr_a3b.json`. Status: post hoc, descriptive.
