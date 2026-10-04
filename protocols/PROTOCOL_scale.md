# Protocole pré-enregistré — le résultat central à plus grande échelle (Qwen2.5-7B, ou Qwen2.5-3B)

Écrit et committé AVANT la sonde et AVANT tout run de la campagne ; sa date est celle de son commit. Les précisions ci-dessous, écrites
avant la sonde, font partie du protocole.

## Cinquième révision de l'engagement
La quatrième révision (`PROTOCOL_exp1_rank16.md`) rompait déjà une formule qui excluait toute révision. Celle-ci en est une cinquième, par
décision de l'auteur, avec une différence déclarée : **le papier est prêt sans ce résultat, et part tel quel s'il n'arrive pas à temps.**
Raison : l'échelle des modèles (1 à 1,5 milliard de paramètres) est la limite la plus visible du papier.

## Précisions, écrites avant la sonde (elles remplacent la version précédente de ces sections)
1. **La sonde** porte sur le bras le plus coûteux en mémoire, `top_unfrozen` au rang 2 (les deux facteurs entraînés ; il construit aussi le
   cache spectral du modèle), taux 2·10⁻³, graine 99, épinglée sur RTX 4000 Ada (20 Go), dans `~/lora-runs-probe`. Elle n'entre dans aucun test.
   **Qwen2.5-7B est retenu si** (a) la sonde se termine, (b) sa mémoire de pointe est inférieure à **18,5 Go**, et (c) la durée totale estimée,
   **105 runs × durée de la sonde ÷ 56** (sept grilles, huit runs chacune à la fois), finit avant la fin de la dernière nuit avant le gel
   (**30/09, 08 h 00**). Sinon, une **seconde sonde, sur Qwen2.5-3B**, avec la même règle ; si elle échoue aussi, **pas de campagne**.
2. **Taux** : 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻².
3. **Deux étapes** : **sélection** en validation sur les graines 0 et 1, à tous les taux (7 bras × 5 taux × 2 graines = **70 runs**) ; puis
   **rapport** sur les graines 2 à 6 au seul taux retenu par bras (**35 runs**), dont les grilles sont générées par la sélection et committées
   avant leurs runs. **L'encadrement au taux oracle n'est donc pas calculable ; il est retiré des analyses rapportées.**
4. Si la campagne n'est pas lue et intégrée avant le gel, **le papier part sans elle** ; elle **n'ouvre aucune autre révision** de l'engagement.

## Montage de la campagne
- OpenBookQA (base : `configs/grids/obqa_r2_common.yaml`, vérifiée), rang 2, le modèle retenu par la règle, épinglé sur RTX 4000 Ada.
- **Bras** : `top`, `bottom` (B gelé) ; `top_unfrozen`, `bottom_unfrozen` (B entraîné) ; `dual_top`, `dual_bottom` (A gelé) ; `free` au rang 1.

## Tests, écrits d'avance (ceux des protocoles existants)
1. **Le gel seul** : `X_unfrozen` − `X`, pour X = `top`, `bottom`, apparié, **t > 3,495**.
2. **Geler A plutôt que B** : `dual_X` − `X`, **t > 3,495**.
3. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), **t > 2,776**.
Rapportés sans issue : G = `free` − `X` ; geler − choisir (bootstrap apparié) ; les taux retenus ; le modèle de carte.

## Engagements
Rapporté quelle que soit l'issue, et quel que soit le modèle retenu par la règle ; aucune graine ni aucun taux ajoutés.
