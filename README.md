# Supplementary material (anonymous submission)

This archive accompanies the submission "Frozen Factor or Spectral Band? Disentangling Two Choices in Low-Rank LoRA". The paper is
self-contained; this material lets a reader check its records, protocols and results.

## Contents

| Path | What it holds |
|---|---|
| `supplementary_record.pdf` | The supplementary record cited by the paper: a cover note, Table R1 (dates of the protocols of the earlier campaigns), and the companion note of an earlier, longer draft, with the full test inventory (Tables S24 to S28). Read its cover note first: its numbering and vocabulary come from the earlier draft. |
| `protocols/` | Every protocol and addendum, as committed before the runs it governs, and three pre-run notes (`NOTE_phase2.md`, `ANALYSE_writes_additif.md`, `EVAL_base_models.md`). `PROTOCOL_DATES.md` gives the date of the first commit of each file in the private repository. Commit hashes are omitted for anonymity. |
| `runs/` | One JSON line per run, gzipped: run identifier, configuration, metrics, status and results row. No weights, checkpoints, saved factors, bases or SVD caches. |
| `code/` | The `lorasub` package (`src/`), the campaign, analysis and figure scripts (`scripts/`), the run grids (`configs/`), tests (`tests/`), the toy notebook (`notebooks/`), and the analysis outputs. |

### Run records

| File | Runs |
|---|---|
| `runs/lora-runs.jsonl.gz` | 6,519 (main campaigns) |
| `runs/lora-runs-tmlr.jsonl.gz` | 681 (campaigns added for this version: boundary, single GPU model, PEFT MiCA, shared recipe, module groups, warm-up seeds) |
| `runs/lora-runs-randA.jsonl.gz` | 53 (A frozen on a random orthonormal basis) |
| `runs/lora-runs-suivi.jsonl.gz` | 45 |
| `runs/lora-runs-v2refs.jsonl.gz` | 30 |
| `runs/lora-runs-traj.jsonl.gz`, `-traj-ctrl` | 10 and 1 |
| `runs/lora-runs-dynamics.jsonl.gz` | 9 |
| `runs/lora-runs-fw.jsonl.gz`, `-p2`, `-probe`, `-warm` | 5, 3, 2 and 2 |

### Analysis outputs in `code/`

- `analyses_tmlr_a1` to `a3c` (`.md` and `.json`): the Holm audit, the minimum detectable effects and the GPU attribution.
- `tmlr_e1.json` to `tmlr_e6.json`, with `tmlr_lecture_e*.txt`: the summaries and readings of the added campaigns (e1 grid boundary,
  e2 single GPU model, e3 PEFT MiCA, e3b shared recipe, e4 module groups, e5 warm-up seeds, e6 random orthonormal A).
- `v2_*_selection.json`, `v2_*_check.json`, `v2_*_basis.json`: selections, checks and basis fingerprints of the earlier campaigns.
- `VERIF_*.md`: the MiCA-equivalence and trainer verifications.
- `analyses_inputs/`: the inputs of the audits, including the earlier draft they were run on.

Some protocols and notes are in French.

## Reproducing

The package installs with `pip install -e .` from `code/` (dependencies in `pyproject.toml` and `requirements.txt`). `code/README.md`
gives the training, grid and aggregation commands. Paths such as `~/lora-shared/data`, `~/lora-runs` and `/tmp/lora-USER` are
specific to the original environment and must be adapted. Exact reproduction of every score also needs the data files and SVD caches,
which are not included.

## Anonymization

- Author names, institutions, user names, cluster, partition and machine names, personal paths and commit hashes are replaced
  (`ANON`, `PARTITION`, `USER`, `<commit>`, `repo`). The git history is not included.
- Internal working notes (planning, drafting and coordination documents) are omitted. Every reported result is in `runs/`, `protocols/`
  and the analysis outputs above.
- Third-party names (cited authors, public repositories) are unchanged.
