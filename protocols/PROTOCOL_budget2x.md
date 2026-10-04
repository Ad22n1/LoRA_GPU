# Protocole pré-enregistré — un second budget d'entraînement (point 5)

Écrit et committé le 25/09, avant tout run.

## Question
L'inversion à taux commun et l'ordre gel/choix tiennent-ils quand on entraîne deux fois plus longtemps ?
Tous les runs de la tâche de format du papier (1 846) sont à 600 000 tokens ; aucun autre budget n'a jamais tourné.

## Montage
Identique à la population principale (Llama-3.2-1B, format, rang 2, sept modules, écrêtage 1,0, α = r, bras aléatoires salés),
sauf le budget : **1 200 000 tokens**, le double. **Les taux ne sont PAS resélectionnés** : ce sont ceux du papier, écrits
explicitement — taux communs 2·10⁻³ et 10⁻² ; taux retenus 10⁻² (`top`, `random`, `random_ortho`), 5·10⁻³ (`bottom`),
10⁻³ (`free` au rang 1). La question porte sur NOTRE configuration entraînée plus longtemps ; un optimum qui se déplacerait avec le
budget est une limite à écrire, pas à corriger après coup. Graines rapportées **2 à 6**.

## Contrôle de sélection au double budget (graines 0 et 1, validation)
Pour chacun des cinq bras, trois taux autour de son taux retenu dans le papier (÷2, ×1, ×2) — 30 runs :
| bras | taux testés |
|---|---|
| `top`, `random`, `random_ortho` (rang 2) | 5·10⁻³, **10⁻²**, 2·10⁻² |
| `bottom` (rang 2) | 2,5·10⁻³, **5·10⁻³**, 10⁻² |
| `free` (rang 1) | 5·10⁻⁴, **10⁻³**, 2·10⁻³ |

Le taux « retenu au double budget » d'un bras est celui des trois qui a la meilleure validation moyenne sur les graines 0 et 1.
**Égalité exacte avec le taux du papier : le taux du papier est conservé.** Un taux retenu à ÷2 ou à ×2 est au bord de cette petite
grille : l'optimum s'est déplacé d'au moins un cran, et on ne dira rien de plus.
- **Si le taux du papier reste le meilleur pour chacun des cinq bras** : la réserve « taux non resélectionnés » est levée.
- **Si un optimum se déplace** : rapporté bras par bras, comme une limite ; les tests principaux restent aux taux du papier, comme prévu.
- **Déplacement du taux (analyse secondaire, prédiction écrite ici)** : au double budget, `bottom` retient un taux strictement plus
  bas que `top`, `random` et `random_ortho`. Issues : **confirmé** (les trois), **partiel** (un ou deux), **non reproduit** (aucun).

## Tests, issues écrites d'avance
Seuils recopiés du protocole des rangs 8 et 16 (`PROTOCOL_rank8_16_format.md`), pour des résultats comparables.

**Inversion** (quatre bras aux deux taux communs) : chaque moitié (l'avance de `bottom` au taux bas, son retard au taux haut) par un
test apparié, **t > 2,776** (bilatéral, 4 d.l.), **contre le meilleur concurrent recalculé par graine ET contre chaque concurrent
fixé** ; interaction bras × taux contre chaque concurrent fixé, **t > 3,96** (Bonferroni sur trois).
1. interaction ET les deux moitiés établies → l'inversion tient au double budget ;
2. interaction établie, une seule moitié OU aucune → partielle, rapportée moitié par moitié ;
3. interaction non établie → l'inversion n'est pas reproduite au double budget : un effet qui dépend du budget.

**Gel contre choix** aux taux retenus, intervalle bootstrap apparié à 95 % :
1. l'intervalle de geler − choisir exclut zéro, positif → l'ordre tient ;
2. il contient zéro → non établi au double budget ;
3. il exclut zéro, négatif → l'ordre s'inverse au double budget.

## Engagements
Rapporté quelle que soit l'issue, avec la réserve des taux non resélectionnés (sauf si le contrôle de sélection la lève) ;
aucune graine ajoutée.
