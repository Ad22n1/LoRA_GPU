# Note à l'écart du point 6 — la cause des écarts de rechargement, trouvée le 26/09

`scripts/diag_reload.py` (sans PARTITION) compare, pour les 20 runs entraînés, le modèle de carte graphique de l'entraînement
(enregistré dans le `results.csv` de chaque run) à celui de l'évaluation rechargée (Slurm) :

| | reproduits | non reproduits |
|---|---|---|
| même modèle de carte | 8 | 0 |
| autre modèle de carte | 1 | 11 |

Les onze runs non reproduits ont tous été réévalués sur un autre modèle de carte ; les huit réévalués sur le même modèle
reproduisent tous (test exact de Fisher, p ≈ 7·10⁻⁵). L'exception (`top`, graine 105) a changé de carte et n'a bougé que d'un
item en validation, dans la tolérance. Le rechargement restaure A et B à l'identique (`load_factors_into`) : la cause est
l'arithmétique bf16 de la génération gloutonne, qui diffère d'un modèle de PARTITION à l'autre, et non le rechargement.

Conséquences : (1) le réentraînement des onze couples, décidé et committé avant ses runs, reste la mesure retenue ;
(2) les évaluations du papier faites dans le processus d'entraînement ne sont pas exposées ; (3) les autres rechargements
(`curvature.py`, `forgetting_from_factors.py`) mesurent des quantités continues, et la remesure de l'oubli reproduit ses
valeurs à 0,03 près.
