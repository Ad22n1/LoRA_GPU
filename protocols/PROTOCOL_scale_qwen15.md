# Protocole pré-enregistré — l'inversion à taux commun, à trois fois l'échelle

Écrit et committé **avant** le lancement.

## Pourquoi
Le premier résultat du papier, l'inversion de l'ordre des bras entre deux taux communs, est mesuré sur
des modèles d'un et d'un demi-milliard de paramètres. Qwen2.5-1.5B est de la même famille que le Qwen du
papier, trois fois plus grand, avec les mêmes réglages de lot : seule la taille change.

## Ce qui est lancé
`bottom`, `top` et `random_ortho` (tirage salé), rang 2, les sept modules, écrêtage 1,0, tâche format,
614 items, aux deux taux communs du papier, 2e-3 et 1e-2, graines 2 à 6 : 30 runs.
Aucun balayage de taux : la mesure porte sur les deux taux communs, comme au §6.

## Ce qui sera calculé, exactement comme au §6
Le basculement de `bottom` = son avance sur le meilleur de `top` et `random_ortho` à 2e-3, plus son
retard sur le meilleur des deux à 1e-2, en points, moyenne sur les graines 2 à 6.

## Ce qui sera conclu, fixé d'avance
- **L'inversion se reproduit** si `bottom` est premier des trois bras à 2e-3 **et** dernier à 1e-2.
- Sinon, elle ne se reproduit pas, et le papier le dira : le premier résultat vaudra alors pour les
  modèles d'un milliard de paramètres et moins.
- Dans les deux cas, le basculement est rapporté avec sa valeur.

## Engagements
- Le résultat est écrit quel qu'il soit, dans le §6, la contribution 1, le résumé et la conclusion.
- Aucune graine ni aucun taux n'est ajouté après coup.
- Un run qui échoue est relancé à l'identique ; une cellule incomplète n'est pas conclue.
