# Protocole pré-enregistré — ARC-Challenge au rang 2

Écrit et committé le 26/09, avant tout run de cette campagne.

## Question
Au rang 16 sur ARC-Challenge, choisir le sous-espace coûte plus que le geler (−5,36 points, [−7,48 ; −3,24]). Aux rangs bas, les
preuves de l'ordre inverse viennent surtout de la tâche de format sur Llama. Le renversement est-il donc un effet du RANG, ou de
la TÂCHE ? Cette campagne mesure la même comparaison sur la même tâche et le même modèle, au rang 2.

## Montage
Identique à ARC-Challenge au rang 32 (`PROTOCOL_arcc_rank32.md`), au rang près : Qwen2.5-1.5B, empreinte `0755bcdbdce3`,
`top`, `bottom`, `random_ortho` (salé) au rang 2 ; `free` au rang 1, qui a 13,4 % de paramètres en plus (1 154 048 contre
1 017 856), comme sur OpenBookQA au rang 2 ; déclaré. Sept taux de 2·10⁻⁴ à 2·10⁻² sur les graines 0 à 6 ; taux choisi sur les
graines 0 et 1 en validation ; rapport sur 2 à 6. Un taux retenu au bord de la grille est rapporté comme une limite.

## Tests, issues écrites d'avance
- **Plancher** : `free` doit dépasser le modèle de base (0,4360) d'au moins 5 points, sinon « non informatif ».
- **Principal — geler − choisir**, aux taux retenus, intervalle bootstrap apparié à 95 % :
  1. **positif, excluant zéro** → au rang 2 sur ARC, geler coûte plus : avec le rang 16, l'ordre change avec le RANG, dans une même
     tâche et un même modèle ;
  2. **contenant zéro** → non établi au rang 2 ; la question rang ou tâche reste ouverte ;
  3. **négatif, excluant zéro** → choisir coûte plus aussi au rang 2 : le renversement est une propriété de la TÂCHE, et le papier
     le dira ainsi.
- **Secondaire, rapporté** : le déplacement du taux ; l'écart du meilleur bras spectral à `random_ortho`.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
