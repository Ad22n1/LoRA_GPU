# Additif au protocole `PROTOCOL_obqa_unfrozen.md` — le modèle de PARTITION

Écrit le 26/09, APRÈS le lancement des runs de cette campagne (le protocole a été committé à 10 h 47, commit `<commit>`, avant tout
run), et AVANT toute lecture de leurs résultats. Il n'en change ni les tests, ni les seuils, ni les issues : il ajoute un contrôle.

## Pourquoi
L'évaluation d'un modèle fixé peut différer de 0,2 à 1,8 point d'un modèle de PARTITION à l'autre (diagnostic du point 6). Cette campagne
compare des runs neufs (`top_unfrozen`, `bottom_unfrozen`) à des références entraînées et évaluées la nuit du 24 au 25/09, peut-être
sur d'autres modèles de carte. Pour le test principal (écarts attendus de 8 à 12 points), l'effet est négligeable ; pour le test du
rétrécissement de l'écart entre bandes (quelques points), il est du même ordre.

## Ce qui est ajouté
1. **Le modèle de PARTITION de chaque run** (lu dans son `results.csv`) est rapporté, bras par bras et graine par graine.
2. **Contrôle** : les tests (1), (2) et (3) sont refaits sur les seules graines où tous les runs comparés ont le même modèle de PARTITION,
   avec le t à n − 1 degrés de liberté ; en dessous de trois graines, le contrôle est déclaré non calculable.
3. **Règle** : si ce contrôle change le verdict du test (3), les deux sont rapportés, et une campagne de suivi est faite : les quinze
   runs de référence (`top` à 5·10⁻³, `bottom` à 2·10⁻³, `free` au rang 1 à 2·10⁻³, graines 2 à 6), qui n'ont pas sauvegardé leurs
   facteurs, sont réentraînés avec leurs facteurs, et tous les runs comparés sont réévalués sur un seul modèle de carte.
   Le verdict pré-enregistré reste celui du protocole ; le suivi est rapporté comme tel.

## Pourquoi pas la réévaluation sur une seule carte, dès maintenant
Elle exigerait les facteurs des runs de référence, qui ne les ont pas sauvegardés : il faudrait d'abord les réentraîner.
