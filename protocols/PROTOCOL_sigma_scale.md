# Protocole pré-enregistré — l'échelle du facteur gelé : Σ, un scalaire, et le rang 4

Écrit et committé **avant** le lancement.

## Ce qui est en jeu
Deux questions, et le titre du papier dépend de la première.

1. **Le rang 4.** Le §8 affirme que l'ordre « le gel coûte plus que le choix » tient au rang 4. Or
   `top_sigma`, qui renverse cet ordre au rang 2, n'a jamais tourné au rang 4. L'affirmation porte
   donc sur une case non mesurée.
2. **Ce que Σ mesure.** Sous Adam, multiplier un facteur **gelé** B par une constante ne change pas
   le pas de A (le gradient est multiplié par la même constante, qu'Adam normalise) mais multiplie
   la mise à jour ΔW = cBA. Donc B = U_r√Σ_r, c'est `top` avec un taux effectif multiplié par √σ,
   direction par direction et module par module. `top_scalar` garde seulement la part par module :
   un scalaire, la moyenne de √σ sur la bande. La forme de Σ à l'intérieur de la bande disparaît.

## Ce qui est lancé
- `top_sigma` au rang 4 : sept taux de 1e-4 à 2e-2, graines 0 et 1 (14 runs), puis les graines 2 à 6
  au taux retenu (5 runs).
- `top_scalar` au rang 2 : idem (14 + 5 runs).
- `top_scalar` au rang 4 : idem, si le budget le permet.
La sélection doit être un maximum intérieur ; sinon la grille est prolongée d'un cran et refaite.

## Ce qui sera conclu, fixé d'avance
**Pour le rang 4.** Un biais doit être neutralisé d'abord : *choisir* est un maximum moins un
minimum, donc ajouter un bras ne peut que l'agrandir, pendant que *geler* diminue si ce bras devient
le meilleur. Les deux mouvements poussent vers « choisir ≥ geler » **avant toute mesure**. Le
protocole fixe donc deux choses :

1. **L'ensemble des bras est figé** : `top`, `bottom`, `random`, `random_ortho`, plus `top_sigma`,
   et rien d'autre. *Choisir* est rapporté à ensemble constant, et aussi sur les quatre bras
   d'origine, pour que les deux valeurs soient comparables à celles des autres rangs.
2. **Trois coûts au node de deux**, parce que `top_sigma` ne change pas la position mais l'échelle,
   et que le compter dans *geler* sans le compter dans *choisir* serait une asymétrie qui sauverait
   le titre :
   - **geler** = `free` (r=2) − le meilleur bras contraint, `top_sigma` compris ;
   - **la position** = étendue entre les quatre bras d'origine, à ensemble constant ;
   - **l'échelle** = `top_sigma` − `top`, le gain que la seule mise à l'échelle du facteur gelé apporte.
   La quantité **primaire** est « geler − position », les deux à ensemble déclaré. La définition
   alternative, où `top_sigma` entre aussi dans *choisir*, est calculée et rapportée à côté ; elle
   n'est pas substituée après coup.
3. **Le taux de `top` au rang 4 est fixé d'avance** : celui de sa propre sélection,
   $2\cdot10^{-3}$, issue du balayage sur 400 items de validation — la seule entorse à la règle des
   sept champs, que le papier signale déjà. Ce choix est **conservateur** : `top` y est plus faible,
   donc *choisir* est plus grand et le titre plus difficile à confirmer. L'autre sélection
   ($5\cdot10^{-3}$, issue du balayage de position sur 614 items) est rapportée à côté.

La quantité décisive est donc « geler − choisir » au rang 4 ainsi défini, avec son intervalle
bootstrap apparié à 95 % (mêmes graines pour tous les bras, la méthode du tableau 4) :
- **l'intervalle exclut zéro par le haut** : l'ordre du titre tient au rang 4, et le rang 2 devient
  une exception documentée, écrite comme telle ;
- **l'intervalle exclut zéro par le bas** : l'ordre « gel avant position » sort du titre et du
  résumé. Le deuxième message devient : ce que coûte le gel dépend de l'échelle du facteur gelé,
  pas seulement de la position ;
- **l'intervalle est entièrement dans $\pm 2$ points** : les deux effets sont déclarés équivalents
  au rang 4, marge fixée ici avant la mesure, et le titre ne les classe plus ;
- **l'intervalle contient zéro sans tenir dans $\pm 2$ points** : le résultat est **non concluant**,
  ce qui n'est pas la même chose qu'une équivalence, et le titre reste inchangé faute de preuve.
Dans les trois cas, le premier message, le taux d'abord, ne bouge pas.

**Cinq graines ne suffisent pas pour trancher un titre.** Un bootstrap sur cinq graines n'a que 126
rééchantillonnages distincts et sa couverture est médiocre. Pour cette quantité seulement, les
graines 7 et 8 sont ajoutées aux cinq cellules impliquées (`top_sigma` r=4, `top` r=4, `bottom` r=4,
`random` r=4, `free` r=2), soit **10 runs de plus**, et l'intervalle est calculé sur sept graines.
Si ces runs ne peuvent pas tourner, l'intervalle à cinq graines est rapporté **avec cette réserve
écrite**, et le titre n'est pas modifié sur sa seule foi. Avant de conclure, on vérifie que les cinq
cellules ont bien les mêmes graines : un appariement incomplet invaliderait l'intervalle.

**Pour `top_scalar` au rang 2**, comparé à `top_sigma` (0,6804) et à `top` (0,6443), graines 2 à 6,
test t apparié bilatéral :
**Ce que l'échelle fait sous Adam, écrit avant de mesurer.** Multiplier un facteur **gelé** B par un
scalaire c laisse le pas de A presque inchangé, puisque le gradient est multiplié par c et qu'Adam le
normalise (à epsilon et à l'écrêtage près), mais multiplie la mise à jour par c. Donc :
- un scalaire **global** ferait de `top_scalar` un simple `top` à un taux décalé, et g mesurerait la
  résolution de la grille, pas le spectre ;
- un scalaire **par module**, celui que ce bras utilise, est un taux par module.
**Le scalaire est fixé avant le lancement** : $s = \lVert\sqrt{\Sigma_r}\rVert_F / \sqrt{r}
= \sqrt{\overline{\sigma}}$, celui qui égalise la norme de Frobenius de `top_sigma`. Tout autre
choix serait un degré de liberté a posteriori.
**Réserve sur l'équivalence avec le taux** : elle n'est pas exacte ici. L'écrêtage global à 1,0 agit
sur la norme *avant* la normalisation d'Adam, et l'epsilon joue pour les petits gradients ; un
facteur d'échelle élevé peut donc déclencher l'écrêtage autrement. Cette réserve est écrite, et le
balayage sans écrêtage sert de contrôle si l'écart entre bras le demande.
**Contrôle de cohérence, sans entraînement** : les scalaires sont lus dans le cache des
décompositions et leur dispersion entre modules est rapportée. S'ils étaient tous égaux, le bras ne
serait qu'un décalage de taux et la comparaison n'aurait pas node d'être.
`top_sigma` contre `top_scalar` isole alors la forme **à l'intérieur** de la bande (σ₁ contre σ₂ au
rang 2), où les valeurs sont souvent proches : un g élevé est donc l'issue attendue, et elle est
annoncée ici pour qu'elle ne soit pas présentée après coup comme une surprise.

Le gain à expliquer est celui de `top_sigma` sur `top` : **+3,61 points** (0,6804 contre 0,6443).
**Le test primaire est l'écart apparié `top_sigma` − `top_scalar`, avec son intervalle** : c'est lui
qui mesure ce que la forme du spectre *dans* la bande apporte, une fois l'échelle du module retirée.
S'il n'est pas distinguable de zéro, l'effet de Σ est attribué à l'échelle par module ; s'il l'est,
la forme compte. Les seuils de 1,2 et 2,4 points sur g restent rapportés comme lecture secondaire,
avec leur intervalle, et l'issue intermédiaire est nommée « partagé » sans attribution.
Soit g = `top_scalar` − `top`, sur les graines 2 à 6, apparié :
- **g ≥ 2,4 points** (au moins deux tiers du gain) : restaurer Σ est un **effet de taux au niveau du
  module**, et le papier rattachera ce résultat à sa thèse principale au node d'en faire une exception ;
- **g ≤ 1,2 point** (moins d'un tiers) : c'est la forme du spectre dans la bande qui compte, et
  c'est un résultat en soi, écrit comme tel ;
- **entre les deux** : les deux parts sont rapportées avec leurs écarts, sans attribution.

## Les cellules en bordure, lancées en même temps
Les grilles `edge_2e2_*` testent 2·10⁻² sur les graines de sélection, pour les cellules qui
sélectionnent aujourd'hui le haut de la grille. **Un optimum ne se déplace que si le gain dépasse
0,005 en validation**, la même marge que pour le taux de `bottom`, afin de ne pas bouger sur du
bruit. Si une cellule change, ses cinq graines rapportées sont relancées au nouveau taux, et les
chiffres recalculés sont fixés ici : le tableau des rangs, le coût du gel de ce rang, le basculement
à taux commun s'il implique cette cellule, et le tableau des départages. Sinon la sélection est
déclarée intérieure.

## Engagements
- Les trois cas sont écrits tels qu'ils sortent, y compris celui qui retire une affirmation du titre.
- Aucune graine ni aucun taux ajouté après coup, hors le prolongement de grille prévu ci-dessus.
- Le mode `top_scalar` est couvert par un test ajouté avec ce protocole ; la suite doit passer avant
  le premier lancement.
- Après ces campagnes, plus aucune n'est ajoutée avant la relecture extérieure et le dépôt.
