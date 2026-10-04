# Protocole pré-enregistré — la tâche de format aux rangs 8 et 16, sur Llama-3.2-1B

Écrit et committé **avant** le premier run, le 24/09.

## Pourquoi
Au rang 16 sur OpenBookQA, l'inversion ne s'est pas reproduite ; mais la tâche, le modèle et le rang y ont changé
ensemble. Cette campagne garde la tâche et le modèle de la population principale (format, Llama-3.2-1B) et ne change
que le rang. Elle dit si l'inversion est un effet de rang ou de tâche.

## Ce qui est identique à la population principale
Llama-3.2-1B, tâche de format, sept modules, écrêtage 1,0, α = r, 600 000 tokens, lot de 2 × 16, séquences de 384,
évaluation sur 614 items avec le vérificateur actuel, tirages salés pour `random` et `random_ortho`.
**Différence déclarée** : pas d'évaluation de l'oubli (le papier ne la mesure qu'au rang 2), pour ne pas ralentir les runs.

## Bras et budget
`top`, `bottom`, `random`, `random_ortho` aux rangs 8 et 16. Référence `free` à +7,5 % du budget, comme aux rangs 2 et 4 :
rang 8 → `free` au rang 4 (2 818 048 contre 2 621 440) ; rang 16 → `free` au rang 8 (5 636 096 contre 5 242 880).

## Sélection
Sept taux, de 2·10⁻⁴ à 2·10⁻², graines 0 et 1, meilleur score de validation. Toute sélection au bord de la grille
entraîne un prolongement d'un cran et une nouvelle sélection. Seconde étape sur les graines rapportées 2 à 6.

## Règles, fixées d'avance
- **Taux communs** : celui que retient `bottom`, et celui du meilleur des autres bras contraints. S'ils sont confondus,
  l'inversion est déclarée **non testable** à ce rang ; aucun autre taux n'est choisi après coup.
- **Tests** : chaque moitié de l'inversion (l'avance de `bottom` au taux bas, son retard au taux haut) par un test apparié,
  seuil bilatéral 2,776, contre le meilleur concurrent recalculé par graine ET contre chaque concurrent fixé ; et le
  **test d'interaction** bras × taux contre chaque concurrent fixé, avec un seuil corrigé de Bonferroni sur ses comparaisons.
- **Gel contre choix** aux taux propres, avec l'intervalle bootstrap apparié, rapporté quelle que soit sa direction.

## Les trois issues, écrites avant les runs (pour chaque rang)
1. **L'inversion disparaît — effet de rang** : `bottom` n'est en tête à aucun des deux taux communs, et l'interaction
   n'est pas établie. La phrase principale du papier devient : l'inversion est un effet des rangs bas.
2. **Elle persiste — effet de tâche** : l'interaction est établie et les deux moitiés aussi. Le résultat du rang 16 sur
   OpenBookQA tient alors à la tâche, pas au rang.
3. **Elle est partielle** : l'interaction est établie mais une seule moitié l'est, ou le classement s'inverse dans les
   moyennes sans que l'interaction soit établie. Rapporté comme tel, sans trancher entre rang et tâche.

## Engagements
Les deux rangs sont rapportés, quelle que soit l'issue. Aucune graine ajoutée, aucun taux choisi après coup.
