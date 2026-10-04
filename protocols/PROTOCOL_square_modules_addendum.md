# Additif à `PROTOCOL_square_modules.md` — budgets exacts et pathologie connue de `q` et `o`

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit. Il ne change ni les bras, ni les taux, ni les graines,
ni les deux tests : il fixe d'avance leur interprétation.

1. **Budgets.** Sur `q_proj` et `o_proj` (2048 × 2048), les bras gelés au rang 2 (B gelé : 2 × 2048 ; A gelé : 2 × 2048) et `free` au rang 1
   (1 × (2048 + 2048)) entraînent exactement **4 096 paramètres par module**. Les trois comparaisons sont à budget rigoureusement égal.

2. **Pathologie connue, déclarée.** Une campagne antérieure (graines 2 à 6) a vu `top`, adapté sur `q` et `o` seuls, s'effondrer : 0,2801 de
   format valide contre 0,7049 pour `free`, avec une perte d'entraînement basse. **Règle d'interprétation fixée ici** : si `top` (B gelé)
   retombe **sous 0,45** de format valide, en moyenne sur les graines rapportées, au taux retenu, le test 1 pour `top` est rapporté mais
   déclaré **non interprétable** pour la question du nombre de paramètres, car il mesurerait cet effondrement ; la conclusion sur le budget
   n'est alors pas tirée. Le seuil est fixé avant tout résultat.

3. **Rapportés en plus.** Le score du modèle de base s'il est enregistré, et, pour chaque bras, toutes les mesures du format enregistrées par
   les runs (clés, nombre, signature…), pour lire un éventuel effondrement.
