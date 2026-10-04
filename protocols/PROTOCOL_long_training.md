# Protocole pré-enregistré — l'isolation au rang 2 avec un entraînement quatre fois plus long

Écrit et committé AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Toutes les campagnes du papier s'entraînent sur le même budget court (600 000 tokens sur la tâche de format). La relecture soupçonne que
certains effets tiennent à cet entraînement court. Cette campagne refait l'isolation du gel de B avec **quatre fois plus de tokens**.

## Montage
- Llama-3.2-1B, tâche de format, rang 2 (base `configs/grids/sigma_residual_C2.yaml`, vérifiée comme la campagne d'origine), avec
  **2 400 000 tokens** au node de 600 000 ; tout le reste identique (lot, accumulation, longueur, sept modules, écrêtage à 1,0).
- **Cinq bras** : `top` et `bottom` (B gelé, r = 2) ; `top_unfrozen` et `bottom_unfrozen` (B entraîné, r = 2) ; `free` (r = 1).
- **Taux re-sélectionnés** (un entraînement plus long déplace l'optimum) : quatre par bras, autour de celui du papier, plus larges vers le bas —
  `top` 2·10⁻³ à 2·10⁻² ; `bottom` 10⁻³ à 10⁻² ; `top_unfrozen` 10⁻³ à 10⁻² ; `bottom_unfrozen` 5·10⁻⁴ à 5·10⁻³ ; `free` 2·10⁻⁴ à 2·10⁻³.
- **Deux étapes** : sélection en validation sur les graines 0 et 1 à tous les taux (**40 runs**), puis rapport sur les graines 2 à 6 au seul
  taux retenu par bras (**25 runs**), dont les grilles sont générées par la sélection et committées avant leurs runs. Un taux retenu au bord est
  une limite, rapportée. Épinglés sur RTX 4000 Ada.

## Tests, écrits d'avance
1. **Le gel seul** : `X_unfrozen` − `X`, au rang 2, pour X = `top`, `bottom`, apparié, **t > 3,495**.
2. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), **t > 2,776** ; lu comme un
   rétrécissement seulement si l'écart gelé vaut au moins 2 points en valeur absolue.
Rapportés sans test : G (`free` − bras gelé), les scores, les taux retenus et leur écart à ceux du papier, le modèle de carte, et la
comparaison aux valeurs à 600 000 tokens.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
