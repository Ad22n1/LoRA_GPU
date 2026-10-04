# Addendum to PROTOCOL_random_A.md — earlier results already read (committed and pushed before any run)

PROTOCOL_random_A.md is not edited. This addendum declares what was known of the arm before the protocol was written.

**Earlier results, read before the protocol.**
1. `PROTOCOL_freeze_A.md` (commit `<commit>`) ran `dual_random_ortho` at rank 2 on the format task with Llama-3.2-1B, alongside `dual_top`
   and `dual_bottom`. Its 49 runs first failed before training (misplaced check, corrected on 26/09); relaunched, they completed, and their
   reading was made on 26/09: the three A-frozen arms lay within 1.34 points of one another, and `dual_random_ortho` was selected at
   2·10⁻², the edge of that grid (read as descriptive).
2. Analysis A3b (results committed in `<commit>`, 03/10, before this protocol) displayed, as coincidental matches of published values, the
   5-seed means of two completed `dual_random_ortho` cells at rank 2: 0.7684 on the format task (rate 2·10⁻², seeds 2 to 6) and 0.5464 on
   OpenBookQA (rate 10⁻², seeds 2 to 6).

**Consequence.** Against the published `random_ortho` (0.6534 and 0.4592), these earlier values made the outcome of P1 largely
foreseeable on both pairs before the protocol was written. P1 is therefore not a blind test: this experiment is a pre-registered
replication, on new runs under a fixed selection rule, of a result already glimpsed. The test, its threshold and its reading are unchanged;
the report states this addendum next to P1. The earlier completed runs are reported descriptively, outside P1 (configuration, seeds,
scores, and when they were read).
