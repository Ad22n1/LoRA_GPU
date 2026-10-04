# Protocole pré-enregistré — le déplacement du taux, sélectionné sur cinq graines

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit (le texte n'en porte pas d'autre).

## Troisième révision de l'engagement — et la dernière
L'engagement du 26/09 a déjà été révisé deux fois (le taux par module, puis la tâche de format sur Qwen), la seconde fois avec la formule
« après elle, aucune campagne avant le gel ». Il est révisé une troisième fois, pour cette seule campagne : elle répond à l'objection la plus
directe contre l'un des trois résultats du papier (le déplacement du taux repose sur une sélection à deux graines), en 120 runs.
**Aucune autre révision ne sera faite : après cette campagne, aucune campagne avant le gel du 30/09, quelle qu'en soit la raison.**

## Question
Le papier affirme, écrit comme une prédiction avant les runs, que `bottom` retient un taux strictement plus bas que les autres bras gelés.
Aux deux configurations qui portent l'affirmation, ce décalage survit-il à une sélection sur cinq graines au node de deux ?

## Montage
- **Deux cellules** : la tâche de format au rang 2 sur Llama-3.2-1B (base : `configs/grids/sigma_residual_C2.yaml`, vérifiée : tâche de
  format, Llama-3.2-1B) ; OpenBookQA au rang 2 sur Qwen2.5-1.5B (base : `configs/grids/obqa_r2_common.yaml`, vérifiée : empreinte
  `dc05f4d847f9`, Qwen2.5-1.5B). Sept modules, écrêtage 1,0, α = r.
- **Bras** : `top`, `bottom`, `random`, `random_ortho`.
- **Graines de sélection** : 0 et 1 (déjà faites, dans la configuration de leur campagne) + **110, 111, 112, neuves** = cinq graines.
  Le sel est activé pour les nouveaux runs des bras aléatoires ; les runs des graines 0 et 1 gardent la configuration de leur campagne.
- **Taux** : 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻² (cinq). **Métrique : la validation seulement**, comme pour toutes les sélections du papier.
- 4 bras × 5 taux × 3 graines neuves = 60 runs par cellule, **120 au total**, épinglés sur les RTX 4000 Ada.

## Prédiction et issues, écrites d'avance
1. **Principal** : sur la moyenne en validation des cinq graines, à chaque cellule, `bottom` retient un taux **strictement plus bas que
   chacun** des trois autres bras. Issues : **confirmé** (plus bas que les trois) ; **partiel** (plus bas qu'un ou deux) ; **non reproduit**
   (plus bas qu'aucun).
2. **Graine par graine** : pour chaque graine, le meilleur taux de chaque bras en validation ; sur combien des cinq graines celui de `bottom`
   est-il strictement plus bas que celui de chacun des trois autres ? C'est ce qui dira si le décalage tient au bruit de la sélection.

## Précisions, écrites avant le commit
1. **Graines de sélection** : 0, 1, 110, 111 et 112, pour les quatre bras, aux deux cellules. Les graines 30 à 32 de `bottom` (tâche de
   format, Llama, qui ont servi au départage du papier) ne comptent pas ; elles sont rapportées à part.
2. **Cellules manquantes des graines 0 et 1** : un taux n'entre dans la comparaison d'une cellule que si ses cinq graines existent pour les
   quatre bras. Les taux exclus sont listés. Aucune cellule manquante n'est lancée.
3. **Égalités** : deux moyennes égales à la quatrième décimale sont rapportées comme une égalité. Le meilleur taux d'un bras est alors un
   ensemble de taux, et « strictement plus bas » n'est atteint que si le plus haut des meilleurs taux de `bottom` est plus bas que le plus
   bas des meilleurs taux de l'autre bras. La même règle vaut graine par graine.
4. **Un bord déjà connu** : sur OpenBookQA, `random_ortho` avait retenu 2·10⁻², le bord haut de cette grille. La prédiction porte sur
   l'ordre des taux, qu'un optimum au-delà du bord ne changerait pas : si l'optimum de `random_ortho` est plus haut encore, `bottom` reste
   plus bas.

## Ce que cette campagne ne change pas
C'est un **contrôle de robustesse de la sélection, pas une nouvelle sélection**. Les tests rapportés dans le papier restent ceux des taux
choisis sur les graines 0 et 1. Si la sélection à cinq graines donne d'autres taux, c'est rapporté comme une limite, sans relancer les tests.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
