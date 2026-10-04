# Protocole pré-enregistré — geler A au rang 16, sur ARC-Challenge et OpenBookQA

Écrit et committé AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Au rang 2, geler A coûte beaucoup moins que geler B. MiCA gèle B et PiCa gèle A, aux rangs 16 et au-delà : ce que le papier peut dire de
leurs comparaisons dépend de ce qu'il en est au rang 16. Les bras gelés (B gelé) et le LoRA libre y existent déjà sur les deux tâches.

## Montage
- Qwen2.5-1.5B, rang 16. **ARC-Challenge** : base `configs/grids/arcc_own_top.yaml` (empreinte `0755bcdbdce3`) ; **OpenBookQA** : la configuration
  du run de référence lui-même (`top`, rang 16, 2·10⁻³, graine 2 ; empreinte `dc05f4d847f9`), pour une configuration identique par construction.
- **Nouveaux bras** : `dual_top` et `dual_bottom` (A gelé sur la bande, B entraîné), au rang 16 ; **sept taux** chacun (ARC-Challenge 5·10⁻⁵
  à 5·10⁻³ ; OpenBookQA 10⁻⁴ à 10⁻²), **toutes les graines à tous les taux** (0 et 1 : sélection en validation ; 2 à 6 : rapport).
  **196 runs**, épinglés sur RTX 4000 Ada.
- **Références existantes** (graines 2 à 6) : ARC-Challenge `top` 5·10⁻⁴ (0,4970), `bottom` 10⁻³ (0,4452) ; OpenBookQA `top` 2·10⁻³,
  `bottom` 10⁻³. La lecture s'arrête si une référence n'a pas ses cinq graines.

## Tests, écrits d'avance (ceux de la campagne du gel de A)
1. **Geler A plutôt que B** : `dual_X` − `X`, au rang 16, pour X = `top`, `bottom`, sur chaque tâche, apparié, **t > 3,495**.
2. **Le rétrécissement de l'écart entre bandes quand A est gelé** : (`dual_bottom` − `dual_top`) − (`bottom` − `top`), **t > 2,776**, lu
   comme un rétrécissement seulement si l'écart gelé vaut au moins 2 points en valeur absolue.
**Contrôle à carte égale, rapporté à côté** : les nouveaux bras tournent tous sur RTX 4000 Ada, alors que les références d'origine
d'ARC-Challenge ont tourné sur plusieurs modèles de carte. Pour ARC-Challenge, les tests (1) et (2) sont aussi calculés contre les références
réentraînées sur RTX 4000 Ada lors de l'étape prévue de l'isolation au rang 16 (`~/lora-runs-suivi`), si elles sont complètes. Pour
OpenBookQA, les modèles de carte des références sont affichés.
Rapportés : les taux retenus, l'encadrement au taux oracle (borne optimiste), le modèle de carte de chaque série.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
