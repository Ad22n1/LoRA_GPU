# Additif au protocole `top_unfrozen` — écrit le 26/09, AVANT tout run de ce bras

`PROTOCOL_top_unfrozen.md` (committé le 22/09, `<commit>`, révisé le 23/09, `<commit>`) n'a jamais été lancé : aucun run
`top_unfrozen` n'existe. Il est exécuté tel qu'il a été enregistré. Cet additif précise quatre points qu'il laissait ouverts ;
il ne change aucun de ses seuils.

## 1. La comparaison au rang 2 reçoit sa règle de test
Le protocole liste `top_unfrozen` (r=2) − `top` (r=2) — le gel seul, au prix d'un budget de 2,15× — sans lui donner de seuil.
Elle est testée par un t apparié bilatéral sur les graines 2 à 6, **t > 3,495** (Bonferroni sur les deux extrémités du point 2) :
établie, le gel coûte à rang et initialisation égaux ; non établie, il ne coûte pas de façon détectable. Son budget n'est pas
apparié, et le papier le dira.

## 2. L'extension à la bande mineure
`bottom_unfrozen` : la base de `bottom` comme B initial, B entraîné, A = 0. Même balayage, mêmes graines, mêmes deux comparaisons :
U_b = `bottom_unfrozen` (r=1) − `bottom` (r=2) contre G_b = `free` (r=1) − `bottom` (r=2), avec la règle 0,7 / 0,3 du protocole ;
et `bottom_unfrozen` (r=2) − `bottom` (r=2), au seuil du point 1. `bottom` est pris à son taux retenu du papier, 5·10⁻³.

## 3. Une seule nuit
Pour tenir avant le gel du 30/09, chaque bras tourne aux sept taux sur les graines 0 à 6 la même nuit. Le taux est choisi
exactement comme le protocole le prévoit — le meilleur en validation sur les graines 0 et 1 — et **seuls les runs des graines 2 à 6
à ce taux sont lus** ; les autres ne servent à rien et ne sont jamais analysés. Si l'optimum tombe au bord, la grille est
prolongée d'un cran la nuit suivante, comme le protocole le prévoit.

## 4. La dérive du sous-espace
Les angles principaux entre le B final et la base de départ sont calculés pour chaque run rapporté, à partir des facteurs
sauvegardés en fin de run, et rapportés avec les comparaisons, comme le protocole l'exige.

## 5. L'incertitude de U, face aux seuils
Les seuils 0,7·G et 0,3·G portent sur des estimations ponctuelles ; avec cinq graines, U peut tomber d'un côté ou de l'autre
par hasard. L'intervalle à 95 % de U (t apparié, 4 d.l.) est donc rapporté, et **si cet intervalle chevauche le seuil qui décide
de l'issue, l'issue est déclarée « partagée »** : U ≥ 0,7·G ne vaut attribution au gel que si la borne basse dépasse 0,7·G ;
U ≤ 0,3·G ne vaut attribution à l'initialisation, à la base ou au rang que si la borne haute reste sous 0,3·G. G est pris à sa
valeur fixée par le protocole (12,41 pour `top`) ; pour `bottom`, à sa valeur mesurée sur les mêmes graines.
La décomposition G = U + (`free` r=1 − `unfrozen` r=1) est rapportée : la première part est ce que dégeler rapporte à
initialisation égale, la seconde ce que l'initialisation spectrale coûte ou rapporte à gel égal. Comme `unfrozen` au rang 1 perd
aussi un rang par rapport au bras gelé au rang 2, U sous-estime plutôt le rôle du gel.

## 6. La décomposition complète, avec un intervalle sur chaque terme
(Ajoutée le 26/09 APRÈS le lancement des runs de cette campagne — lancés à partir du commit `<commit>` —, mais AVANT toute
lecture de leurs résultats. Elle est purement descriptive : elle ne change ni U, ni aucune règle, ni aucune issue.)
G = [`free` r=1 − `unfrozen` r=1] + [`unfrozen` r=1 − `unfrozen` r=2] + [`unfrozen` r=2 − gelé r=2]
Le premier terme est l'initialisation à rang et budget égaux, le deuxième le rang (et le budget) à gel et initialisation égaux,
le troisième le gel seul à rang et initialisation égaux. Chacun est rapporté avec son intervalle à 95 % (t apparié, graines 2 à 6).
Cette décomposition est descriptive ; elle ne change ni U, ni la règle du point 5.

## Engagements
Ceux du protocole : tout est rapporté quel que soit le sens ; aucune graine ni aucun taux ajoutés, hors le prolongement prévu.
