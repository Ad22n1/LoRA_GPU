# Protocole pré-enregistré — geler A contre geler B à nombre de paramètres égal (modules carrés)

Écrit et committé AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Geler B entraîne A (r·d_in paramètres par module) ; geler A entraîne B (r·d_out). Sur les sept modules, les bras qui gèlent A entraînent
11 à 27 % de paramètres en plus selon le modèle. L'avantage de geler A plutôt que B pourrait donc tenir en partie au nombre de paramètres.
Sur les modules **carrés**, `q_proj` et `o_proj` de Llama-3.2-1B (2048 × 2048), les deux bras entraînent exactement le même nombre de
paramètres, par construction : 4 096 par module au rang 2.

## Montage
- Llama-3.2-1B, tâche de format, rang 2, **seulement `q_proj` et `o_proj`** ; base `configs/grids/sigma_residual_C2.yaml`, vérifiée comme la
  campagne d'origine (600 000 tokens, longueur 384, lot 2, accumulation 16, écrêtage 1,0).
- **Cinq bras** : `top`, `bottom` (B gelé, r = 2) ; `dual_top`, `dual_bottom` (A gelé, r = 2) ; `free` (r = 1).
- **Taux re-sélectionnés** (deux modules au node de sept) : cinq par bras — bras gelés 2·10⁻³ à 5·10⁻² ; `free` 5·10⁻⁴ à 10⁻².
- **Graines neuves** : 130 et 131 pour la sélection (validation), 140 à 144 pour le rapport, au seul taux retenu par bras ; les grilles du
  rapport sont générées par la sélection et committées avant leurs runs. Un taux retenu au bord est une limite, rapportée. **75 runs**,
  épinglés sur RTX 4000 Ada. (Les graines 2 à 6 sont évitées : une campagne antérieure sur `q` et `o` les a utilisées.)

## Tests, écrits d'avance (ceux de la campagne du gel de A)
1. **Geler A plutôt que B, à paramètres égaux** : `dual_X` − `X`, pour X = `top`, `bottom`, apparié, **t > 3,495**.
2. **Le rétrécissement de l'écart entre bandes quand A est gelé** : (`dual_bottom` − `dual_top`) − (`bottom` − `top`), **t > 2,776**, lu
   comme un rétrécissement seulement si l'écart gelé vaut au moins 2 points en valeur absolue.
**Lecture fixée d'avance** : la conclusion sur le nombre de paramètres repose sur la bande dominante (`top`). Le bas du spectre de `q` et `o`
est dégénéré sur Llama (valeurs singulières quasi nulles) : la bande mineure y est atypique, et ses résultats sont rapportés avec cette réserve.
Rapportés sans test : G, les scores, les taux retenus, le modèle de carte.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
