# Analyses for the TMLR version — definitions, committed before execution

Part A of the TMLR additions. **No new run.** These analyses read results the paper already reports; they change no published result.
Status of every output: **post hoc · descriptive** (the definitions are fixed here, before execution, but the data exist).
Inputs are versioned with this file: `analyses_inputs/companion_tmlr.tex` (the companion note of the TMLR version, the source of the test
tables) and `analyses_inputs/tests_spec.json` (the transcription of its 145 tests; see A1). Script: `scripts/analyses_tmlr.py`.

## A1 — Holm over all 145 tests

**Population.** The 145 tests of the paper are the 151 rows of the companion's Tables `tab:tests` to `tab:tests4`, minus the four
descriptive rows (D: rows 8, 9, 11, 119) and the two selection outcomes (C: rows 61, 84), as the paper counts them. Rows are numbered
in table order, 1 to 151. A row that covers several quantities counts once, as in the paper.

**Transcription.** `tests_spec.json` gives, for each row, the statistics as printed (t values, intervals, p-values), the degrees of
freedom (reported seeds minus one), the sidedness stated by the row's threshold (or implied by its critical value), the threshold where
the row tests an interval against one (0.7 G, 0.3), and which of the row's components the paper calls established. Values that a row
takes from another table are taken from that table: rows 1, 4 and 12 from `tab:bandqwen`; 0.7 G for rows 85, 92, 114, 130 and 131 from
the G stated for that population (rank-16 isolation table, `tab:headline`, the fresh-seed table and `tab:rankc`). Each such source is
named in the row's `note`.

**From a row to a p-value**, by class (each rule applies to every component of the row):
- `t` — paired t statistic with ν degrees of freedom. Two-sided rows: p = 2·P(T_ν > |t|). One-sided rows: p = P(T_ν > t). Rows whose
  prediction had a sign and whose statistic has the opposite sign (rows 56 and 58, « NE » with |t| above the threshold) get p = 1.
  A range « t = a to b » is transcribed as its two ends.
- `ci`, bootstrap 95 % interval against 0 — normal approximation on the side of zero: estimate e (as printed, or the midpoint),
  SE = (e − lower)/1.96 if e ≥ 0, else (upper − e)/1.96; p = 2·(1 − Φ(|e|/SE)). **This is an approximation** (the bootstrap interval is not
  normal); rows near the Holm threshold under it are flagged.
- `ci`, t-based 95 % interval against a threshold c (R against 0.7 G; P against 0.3) — e = midpoint, SE = (upper − lower)/(2·t_{0.975,ν});
  one-sided p = P(T_ν > (e − c)/SE) for « above c », P(T_ν > (c − e)/SE) for « below c ». Row 16 (t interval against 0) is two-sided.
- `cis` — one bootstrap interval per component (factor, band, interaction), each by the bootstrap rule.
- `tost` — equivalence within ±m from a 90 % interval with ν degrees of freedom: e = midpoint, SE = (upper − lower)/(2·t_{0.95,ν}),
  p = max(P(T_ν > (e + m)/SE), P(T_ν > (m − e)/SE)).
- `conj` — a reading that requires two criteria (rows 126, 127, 144: lower bound above 0.7 G and t above its threshold): p = max of the two.
- `ci2` — row 148, inconclusive between « suffices » (P > 0.7) and « does not suffice » (P < 0.3): p = the smaller of the two one-sided p.
- `p` — the printed p-value (row 2 prints « p ≥ 0.016 »: 0.016 is used).
- `none` — a criterion that is not a statistical test (a floor checked in advance, a row below its floor, a row not read): it keeps its
  place among the m = 145 tests with p = 1, and its own status is reported as **outside Holm** (it carries no p-value to correct).

**One p-value per row.** If the paper calls some components of the row established, the row's p is the **largest** p among those
components (all of them must survive together). Otherwise it is the smallest p among its components.

**Holm.** Step-down Holm at family-wise α = 0.05 over m = 145: sort the p-values, adjusted p_(i) = max_{j ≤ i} min(1, (m − j + 1)·p_(j)).
An established result **stays established** if its row's adjusted p ≤ 0.05. Reported: for every row, its p, its Holm-adjusted p, the
paper's status and the status after Holm; the list of established results that stay and of those that fall; the rows flagged as
approximate (bootstrap rule) whose adjusted p lies between 0.01 and 0.1.

**Check of the transcription.** Before Holm, each row's p is compared with the row's own threshold (its uncorrected or within-family
level, from its critical value and family size). Every row whose conversion does not reproduce the paper's own verdict is listed; such a
row is not corrected silently: the list says whether the cause is the transcription or the approximation.

## A2 — Minimum detectable effect of the band comparisons reported as not established

**Comparisons.** Each band comparison the paper reports as not established — a difference between two bands or band positions, or
between their gaps — taken as the components of these rows: 1 (β = 0.75 against β = 0 and β = 0.5), 3, 4 (against β = 0.5), 5, 10, 12
(against β = 1), 14, 17, 18, 21, 26, 35, 42, 48, 56, 58, 110, 125 (band component), 138, 142.

**Definition.** For a paired t-test on n seeds (ν = n − 1) at the row's own level α (two-sided or one-sided as in the row, including its
family correction), the minimum detectable effect at 80 % power is MDE = δ_z · σ_d, where δ_z solves power(δ_z) = 0.80 under the
non-central t distribution with non-centrality δ_z·√n, and σ_d is the observed standard deviation of the paired differences over the
reported seeds. σ_d = |d̄|·√n / |t| when the paper states both the mean difference d̄ and t; otherwise σ_d is computed from the runs of
the two compared cells, identified by their published values as in `scripts/analyses_v3.py`. Reported: n, α, δ_z, σ_d (with its
source), the MDE in points, and the observed difference.

## A3 — PARTITION model behind each reported cell

**Cells.** Every cell whose score the main paper reports in a table, identified in the runs by its published value (as in
`scripts/analyses_v3.py`, ties settled by validation as the selection does).

**Measure.** For each cell, the PARTITION model recorded by each of its runs (the run's own record; the field is located by name and its
location is reported), counted per model; runs with no recorded PARTITION model are listed by identifier. Reported: one row per cell — task,
model, arm, rank, rate, seeds, PARTITION models with counts, and the runs without a record.

## Rules common to A1-A3
Reported whatever the outcome. A missing input is stated, never replaced silently. Any departure from these definitions after commit is
declared as a deviation, with its commit.
