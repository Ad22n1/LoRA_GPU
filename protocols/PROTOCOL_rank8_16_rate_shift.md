# Additif 2 au protocole des rangs 8 et 16 — la prédiction sur le déplacement du taux

Écrit et committé le 24/09 au soir, **après le lancement de la sélection (17 h) et AVANT la lecture de tout résultat** de cette
campagne. Aucun run de plus : il ajoute une prédiction sur des taux que la sélection choisit de toute façon.

## Pourquoi
Le déplacement du taux — la bande mineure choisit un taux plus bas — est devenu le résultat le plus général du papier
(rang 2 sur Llama et Qwen, rang 16 sur OpenBookQA). ARC-Challenge vient d'en fournir un contre-exemple : `bottom` y retient
10⁻³, plus haut que `top` (5·10⁻⁴) et égal à `random_ortho`. La question devient : ce déplacement est-il général, ou
dépend-il de la tâche et du rang ? Cette campagne y répond sur la tâche de format, sans changer de modèle.

## La prédiction, pour chaque rang (8 et 16), sur la sélection telle que le protocole la définit
**`bottom` choisit un taux strictement plus bas que `top`, que `random` et que `random_ortho`.**
« Plus bas » s'entend sur la grille : un taux égal n'est pas plus bas. Une sélection au bord est d'abord prolongée d'un cran,
comme le protocole le prévoit, et c'est le taux final qui compte.

## Les issues, pour chaque rang
1. **Confirmée** : le taux de `bottom` est strictement plus bas que celui de chacun des trois autres bras gelés.
2. **Partielle** : plus bas que celui d'un ou deux d'entre eux seulement.
3. **Réfutée** : plus bas qu'aucun.

## Engagements
Rapportée pour les deux rangs, quelle que soit l'issue, à côté du résultat sur l'inversion. Le papier dira où le déplacement
apparaît et où il n'apparaît pas — sur les tâches (format, OpenBookQA, ARC) comme sur les rangs.
