# Protocole — l'effet du taux au rang 4 (Llama) et au rang 2 (Qwen)

Écrit et committé **avant** le lancement, le 21/09 au soir.

## Pourquoi
La figure de la thèse (fig. 1) mesure l'effet du taux au rang 2 sur Llama seulement ; au rang 4 elle
dit « not measured », et sur Qwen l'inversion à taux commun n'a jamais été mesurée.

## Ce qui est lancé (graines 2 à 6, les graines rapportées, taux fixés dans les grilles)
- Llama, rang 4 : `top` et `bottom` à 1e-2, `random` à 2e-3 (15 runs). Existent déjà : `top` et
  `bottom` à 2e-3, `random` à 1e-2.
- Qwen, rang 2 : `top`, `random_ortho`, `bottom` à 2e-3, `bottom` à 1e-2 (20 runs). Existent déjà :
  `top` et `random_ortho` à 1e-2.
- Llama, rang 2 : `random` à 2e-3 graine 5 (1 run), pour la case vide du tableau de l'oubli.

## Ce qui sera calculé, exactement comme pour la figure 1
Le basculement de `bottom` = son avance sur le meilleur des autres bras à 2e-3 **plus** son retard sur
le meilleur des autres à 1e-2, en points de `format_parsed`, moyenne sur les graines 2 à 6.
Même population que les tableaux (7 modules, écrêtage 1,0, pas de recalage, 614 items, tirages salés).

## Ce qui sera écrit, quel que soit le résultat
- Au rang 4, la barre « taux » de la figure 1 portera la valeur mesurée. **Si elle est inférieure au coût
  du gel au rang 4, le texte dira qu'au rang 4 le gel passe avant le taux**, et le titre restera
  justifié par le rang 2 seulement, ce que le texte dira aussi.
- Sur Qwen, on écrira si l'ordre des bras s'inverse entre 2e-3 et 1e-2, avec sa valeur, qu'il
  s'inverse ou non.
- Aucune graine ne sera ajoutée après celles-ci pour ces deux mesures.
