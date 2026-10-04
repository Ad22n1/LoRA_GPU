# Écart au protocole « position ou norme » (point 6) — écrit le 26/09, AVANT le lancement des runs qu'il ajoute

## Ce qui a été constaté
Le protocole prévoyait d'entraîner chaque couple (bras, graine) une fois, puis de recharger ses facteurs et de l'évaluer aux sept
normes, après avoir vérifié que l'évaluation rechargée reproduit le score enregistré (à trois items près). **Onze couples sur vingt ne
reproduisent pas** : `bottom` 106 et 107 ; `random` 106, 107 et 108 ; `random_ortho` 105 à 109 (les cinq) ; `top` 107. Écarts de
0,2 à 1,8 point. Le script s'est arrêté pour eux, sans rien mesurer : ils n'ont aucune évaluation aux sept normes.
La cause n'est pas établie. `random_ortho` échoue à chaque fois et les bras aléatoires sont les plus touchés, ce qui oriente vers la
reconstruction de la matrice figée B au rechargement ; les échecs isolés de `top` et `bottom` peuvent venir de l'évaluation elle-même.

## Ce qui est décidé, avant tout run
Pour ces onze couples seulement, les sept normes sont mesurées **sans rechargement** : chaque couple est réentraîné avec la norme cible
appliquée par `rescale_update_` dans le même processus, juste après l'entraînement — la méthode des cibles 1,5 et 4,5 existantes.
Même configuration, mêmes graines, mêmes taux (10⁻² pour `top`, `random`, `random_ortho` ; 5·10⁻³ pour `bottom`) : 11 × 7 = 77 runs.
- Chaque couple a donc, selon le cas, ses sept normes mesurées par RECHARGEMENT (9 couples) ou par RÉENTRAÎNEMENT (11 couples) ;
  l'analyse le signale couple par couple, et donne l'issue avec les vingt couples, puis avec les neuf seuls.
- Le test, les seuils, la cible utilisable et les quatre issues restent ceux du protocole.
- Nuance déclarée : pour les onze couples réentraînés, la « même direction à sept amplitudes » n'est qu'approchée — sept entraînements
  distincts d'une même configuration, que le calcul sur carte graphique ne rend pas bit à bit identiques.
