# Note à `PROTOCOL_freeze_A.md` — le bras `dual_random_ortho` : un défaut de code, corrigé

Écrite le 26/09, APRÈS la lecture des résultats principaux de ce protocole (issue 1 aux deux extrémités), et AVANT la relance du bras.

## Ce qui s'est passé
Les 49 runs de `dual_random_ortho` (7 taux × graines 0 à 6) ont tous échoué **avant tout entraînement**, sur la même erreur :
`ValueError: mode 'dual_random_ortho' needs an SVDEntry of the base weight`. Ce bras tire une base orthonormée aléatoire et n'utilise
aucune décomposition ; `inject_lora` n'en charge donc aucune pour lui, mais le constructeur des bras qui gèlent A vérifiait la présence
d'une décomposition pour TOUS ces bras, avant de regarder lequel était construit.

## La correction
Dans `LoRALinear`, les vérifications de la décomposition ne portent plus que sur les bras qui utilisent Vᵀ (`dual_top`, `dual_bottom`,
`dual_random`), dont le code ne change pas ; `dual_random_ortho` garde une vérification du rang. Un test,
`test_dual_random_ortho_needs_no_svd`, construit le bras sans décomposition et vérifie ses propriétés (A orthonormée et gelée, B nul au
départ, budget r × d_out, modèle inchangé au pas 0, une base par module grâce au sel, la même base pour la même graine).

## Conséquences
- Le test principal du protocole (`dual_top` et `dual_bottom` contre le gel de B) n'est pas touché : ces bras ont tourné normalement.
- `dual_random_ortho` ne sert qu'à l'analyse secondaire, descriptive (l'étendue entre les bras qui gèlent A). Il est relancé à
  l'identique (même grille, même configuration) APRÈS la lecture des résultats principaux ; il est rapporté comme tel.
- La campagne `PROTOCOL_obqa_freeze_A.md`, committée mais pas encore lancée, part avec le code corrigé.
