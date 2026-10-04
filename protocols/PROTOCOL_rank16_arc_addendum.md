# Additif au protocole ARC — écrit le 24/09, après la sélection, avant tout run sur les graines rapportées

Ce que la sélection (graines 0 et 1, validation) a donné, et ce que le protocole ne prévoyait pas.

## ARC-Easy : deux sélections au bord
`top` et `free` retiennent 10⁻⁴, le bas de la grille. Comme le protocole le prévoit, la grille est
prolongée d'un cran (5·10⁻⁵) pour ces deux bras, sur les graines 0 et 1, et la sélection est refaite.
Aux taux bas, les bras sont à un point les uns des autres en validation ; le modèle de base fait déjà
0,711 en test, donc le plancher de cinq points sera serré. Les deux sont rapportés tels quels.

## ARC-Challenge : la règle ne donne qu'un seul taux commun
`bottom` retient 10⁻³ et le meilleur des autres bras contraints, `random_ortho`, aussi 10⁻³. Les deux
taux communs sont confondus : **le test d'inversion n'est pas défini sur ce benchmark**, et il est
rapporté comme tel. Aucun autre taux n'est choisi après coup pour le rendre testable.
Ce qui reste mesuré, sur les graines 2 à 6 : le test de plancher, le gel contre le choix aux taux
propres, et, à titre descriptif, les trois bras au même taux (10⁻³), sans test d'inversion.

## Le déplacement du taux
Sur les deux benchmarks ARC, `bottom` ne retient pas un taux plus bas que `top`. Avec des écarts de cet
ordre en validation, le papier le rapporte comme une absence de déplacement, pas comme un déplacement
inverse.
