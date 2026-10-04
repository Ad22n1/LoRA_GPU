# Protocole pré-enregistré — le résultat central sur une troisième famille : Mistral-7B-v0.3

Écrit et committé AVANT la sonde et AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Le résultat central tient sur Llama-3.2-1B, Qwen2.5-1.5B et Qwen2.5-7B : deux familles, une seule à grande échelle. Mistral-7B-v0.3 (7,2
milliards de paramètres, Apache 2.0, utilisé par LoRA-XS) est une troisième famille, à l'échelle du 7B.

## Montage : celui de `PROTOCOL_scale.md`, sans modèle de repli
1. **Sonde** : `top_unfrozen` au rang 2, taux 2·10⁻³, graine 99, épinglée sur RTX 4000 Ada, dans `~/lora-runs-probe` ; hors de tout test.
   **Mistral-7B est retenu si** la sonde se termine, que sa mémoire de pointe est inférieure à 18,5 Go et que la durée estimée
   (105 runs × durée de la sonde ÷ 56) finit avant le 30/09 à 8 h. **Sinon, pas de campagne** : il n'y a pas de modèle de repli.
2. **Sélection** en validation sur les graines 0 et 1, à cinq taux (5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻²), puis **rapport** sur les graines 2
   à 6 au seul taux retenu par bras, les grilles du rapport étant générées par la sélection et committées avant leurs runs.
- OpenBookQA (base `configs/grids/obqa_r2_common.yaml`, vérifiée), rang 2. Bras : `top`, `bottom` (B gelé) ; `top_unfrozen`,
  `bottom_unfrozen` (B entraîné) ; `dual_top`, `dual_bottom` (A gelé) ; `free` au rang 1. **105 runs**, épinglés sur RTX 4000 Ada.
- Un run en échec pour une cause de machine (mémoire, disque, environnement) est relancé une fois à l'identique, sur les machines où un run
  de Mistral s'est terminé ; au-delà, son bras est déclaré incomplet.

## Tests, écrits d'avance (ceux du 7B)
1. **Le gel seul** : `X_unfrozen` − `X`, pour X = `top`, `bottom`, **t > 3,495**.
2. **Geler A plutôt que B** : `dual_X` − `X`, **t > 3,495**.
3. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), **t > 2,776**.
Rapportés sans test : G, « gap minus spread », les taux retenus, le modèle de carte. Un taux retenu au bord est une limite, rapportée.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
