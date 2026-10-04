# Protocole pré-enregistré — ARC-Challenge au rang 32

Écrit et committé le 26/09, avant tout run de cette campagne.

## Question
Au rang 16, sur ARC-Challenge, choisir le sous-espace coûte plus que le geler (−5,36 points, [−7,48 ; −3,24], pré-enregistré).
Cet ordre tient-il au rang 32, plus proche des rangs où les variantes spectrales sont évaluées ?

## Montage
Qwen2.5-1.5B, ARC-Challenge (empreinte des données `0755bcdbdce3`), sept modules, écrêtage 1,0, α = r, 600 000 tokens.
`top`, `bottom`, `random_ortho` au rang 32 (bras aléatoire salé, écrit explicitement) ; `free` au rang 14, qui a 0,79 % de
paramètres en moins (16 156 672 contre 16 285 696), comme au rang 16.
- **Sélection et rapport, en une seule nuit** : chaque bras tourne aux sept taux 5·10⁻⁵ à 5·10⁻³ sur les graines 0 à 6 ; son taux
  est le meilleur en validation sur les graines 0 et 1 SEULEMENT ; le rapport se lit sur 2 à 6, à ce taux. Un taux retenu au bord
  de la grille est rapporté comme une limite.

## Tests, issues écrites d'avance
- **Plancher** : `free` doit dépasser le modèle de base (0,4360) d'au moins 5 points, sinon « non informatif ».
- **Principal — geler contre choisir**, aux taux retenus, intervalle bootstrap apparié à 95 % :
  1. geler − choisir exclut zéro, NÉGATIF → choisir coûte plus, comme au rang 16 : l'ordre inversé tient au rang 32 ;
  2. contient zéro → non établi au rang 32 ;
  3. exclut zéro, POSITIF → l'ordre revient à celui des rangs bas.
- **Secondaire, rapporté** : le déplacement du taux (`bottom` plus bas que `top` et `random_ortho`, ou non). L'inversion à taux
  commun n'est pas testée dans cette campagne.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
