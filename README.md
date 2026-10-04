# Code

The `lorasub` package and the scripts behind the paper. Paths such as `$SHARED`, `~/lora-shared` and `/tmp/lora-USER` come from the
original cluster environment and must be adapted.

## Install and test

```bash
pip install -r requirements.txt && pip install -e .
pytest -q          # CPU tests with a toy model and a character tokenizer, no download
```

## Data

```bash
python -m lorasub.data.format OUT --n_train 5000 --n_val 500 --n_test 1000 --seed 0 --variant transform   # synthetic format task
bash scripts/setup_data.sh                                                                               # data and SVD caches
```

OpenBookQA and ARC are loaded from their public releases (`src/lorasub/data/commonsense.py`).

## One run, a grid, and the aggregation

```bash
python -m lorasub.train configs/example_facts.yaml mode=top seed=1
python -m lorasub.launch_grid configs/grids/<grid>.yaml --grid_dir $SHARED/grid/<name> --partition PARTITION --max_par 120
python -m lorasub.aggregate --runs_dir $SHARED/runs --out $SHARED/analysis --select_lr configs/lr_selected.yaml
```

`launch_grid` writes one YAML per run and a SLURM array; `aggregate` selects each configuration's learning rate on validation and builds
the tables. `cache_svd.py` builds the SVD caches, `grad_probe.py` the initial-gradient diagnostics, `mica_peft.py` the PEFT MiCA arm.

## Where the paper's campaigns are

- `configs/grids/`: the grid of every campaign.
- `scripts/`: campaign drivers, audits and figure scripts; `scripts/analyses_tmlr.py` runs analyses A1 to A3.
- The run records are in `../runs/`, the protocols in `../protocols/`.
