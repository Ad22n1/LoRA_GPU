# Protocole pré-enregistré — réplication du rang 16 sur ARC-Easy et ARC-Challenge

Écrit et committé **avant** le lancement, le 24/09, après avoir lu le résultat d'OpenBookQA.

## Pourquoi
Sur OpenBookQA au rang 16, l'inversion ne s'est pas reproduite : `bottom` est dernier aux deux taux
communs, son taux sélectionné est le plus bas des bras contraints, et le gel coûte 2,1 points. Un
seul benchmark ne dit pas si c'est une propriété du rang ou de la tâche. ARC-Easy et ARC-Challenge
font partie de la même suite de sens commun et répondent à cette question.

## Ce qui est identique à OpenBookQA
Qwen2.5-1.5B, rang 16, sept modules, écrêtage 1,0, α = r ; `top`, `bottom`, `random_ortho` au rang 16
et `free` au rang 7 (−0,8 % de paramètres) ; balayage de sept taux (10⁻⁴ à 2·10⁻²) sur les graines 0
et 1 ; sélection sur la validation ; rapport sur les graines 2 à 6 ; évaluation par vraisemblance des
options. Différence déclarée : ARC-Easy a 2 251 items d'entraînement et ARC-Challenge 1 119, contre
4 957 ; au même budget de 600 000 tokens, le modèle les voit donc plus de fois.

## Règles, fixées d'avance
- **Plancher** : `free` doit dépasser le modèle de base d'au moins 5 points ; sinon, « non informatif ».
- **Taux communs** : celui que retient `bottom` et celui du meilleur des autres bras contraints.
- **Inversion** : chaque moitié passe son test apparié (seuil bilatéral 2,776), **contre le meilleur
  concurrent recalculé par graine et contre chaque concurrent fixé séparément** (demande de la
  relecture TMLR). Une moitié non établie est dite non établie.
- **Gel contre choix** aux taux propres, avec l'intervalle bootstrap apparié, rapporté quelle que soit
  sa direction.
- **Déplacement du taux** : rapporté comme sur OpenBookQA.

## Engagements
- Les deux benchmarks sont rapportés, quel que soit leur résultat, avec OpenBookQA.
- Aucun taux commun choisi après coup ; aucune graine ajoutée.
