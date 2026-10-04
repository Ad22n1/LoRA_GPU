# Protocole pré-enregistré — reproduire le diagnostic : B gelé sur la direction apprise

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : mardi 29/09 a 20 h 00 (2026-09-29 20:00).** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Pourquoi
La phase 2 (OpenBookQA, Qwen2.5-1.5B, rang 2, direction de la graine 0) conclut que l'essentiel du coût du gel de B vient de l'endroit où B
est gelé (P = 1,16 [1,05 ; 1,28]). Elle repose sur une paire et une graine de direction. Ce protocole la reproduit sur une deuxième graine de
direction et sur une deuxième paire. Il part sans condition.

## Montage, identique à la phase 2 sauf ce qui est indiqué
- Réplique A — même paire, autre direction : OpenBookQA, Qwen2.5-1.5B, rang 2. La direction vient d'un LoRA libre au rang 2, graine 7, au
  taux 2·10⁻³, facteurs sauvegardés (la graine 7 n'entre dans aucun score rapporté sur cette paire). G = 8,04, fixé d'avance.
- Réplique B — autre paire : tâche de format, Llama-3.2-1B, rang 2. La direction vient d'un LoRA libre au rang 2, graine 0, à son taux
  retenu (2·10⁻³), facteurs sauvegardés. G = 12,41 (bande dominante, `top`), fixé d'avance.
- Pour chaque module, la base du bras `learned` est la base orthonormale de l'espace des colonnes du B de ce LoRA libre, obtenue par QR ;
  B gelé sur cette base, sans Σ ; A entraîné depuis zéro ; même budget que `top`.
- Taux : cinq (les cinq de la phase 2 : 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻²), sélection en validation sur les graines 0 et 1, rapport sur
  les graines 2 à 6 au taux retenu, un cran de prolongement si l'optimum tombe au bord.
- Références publiées, graines 2 à 6 : `top`, `bottom`, `random_ortho`, `free` (r = 1). Si une référence n'a pas tourné entièrement sur
  RTX 4000 Ada, elle est réentraînée à l'identique sur ce modèle.
- Une seule carte, RTX 4000 Ada. Environ 32 runs, 5 à 6 PARTITION-heures.

## Test et lecture, écrits d'avance (pour chaque réplique)
`learned` moins `top`, apparié par graine, t > 2,776, rapporté avec son intervalle à 95 % et P = (`learned` − `top`) / G.
- Si la borne basse de l'intervalle atteint 0,7·G : l'essentiel du coût vient de l'endroit où B est gelé.
- Si la borne haute reste sous 0,3·G, que le test soit établi ou non : l'endroit compte peu.
- Sinon : non résolu.
Rapportés sans test : `learned` moins `bottom`, moins `random_ortho` et moins `free`.

## Ce que le papier en dira
Si les deux répliques concluent comme la phase 2, le résumé pourra dire « on two task–model pairs and two direction seeds ». Sinon, il
rapportera chaque réplique telle qu'elle tombe, et gardera la formulation actuelle, limitée à la paire testée.

## Règles communes
Une relance pour échec machine ; rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés hors le cran de prolongement.
