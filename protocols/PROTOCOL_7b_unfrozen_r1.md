# Protocole pré-enregistré — B entraîné au rang du LoRA libre, sur Qwen2.5-7B et Mistral-7B

Écrit et committé AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Sur les deux modèles de 7 milliards de paramètres, le coût isolé du gel de B est établi à rang égal (rang 2), mais B n'a pas été entraîné au
rang de la référence libre (rang 1) : la comparaison à budget égal — le test U de l'isolation — y manque.

## Montage
- OpenBookQA, base `configs/grids/obqa_r2_common.yaml` (vérifiée), le modèle seul remplacé, **exactement comme les deux campagnes 7B**.
- **Deux bras par modèle** : `top_unfrozen` et `bottom_unfrozen` **au rang 1** ; mêmes cinq taux (5·10⁻⁴ à 10⁻²).
- **Deux étapes** : sélection en validation sur les graines 0 et 1 (40 runs), puis rapport sur les graines 2 à 6 au seul taux retenu par bras
  (20 runs), dont les grilles sont générées par la sélection et committées avant leurs runs. Un taux retenu au bord est une limite, rapportée.
- **60 runs**, épinglés sur les machines où un run du même modèle s'est déjà terminé (RTX 4000 Ada). Un run en échec pour une cause de
  machine est relancé une fois à l'identique sur ces machines ; un run laissé non démarré par le code (carte occupée) part au passage suivant.
- **Références existantes** (graines 2 à 6), et **G fixés d'avance** : Qwen2.5-7B `top` 0,5316 (5·10⁻³), `bottom` 0,4840 (5·10⁻⁴), `free`
  (r = 1) 0,6284, **G = 9,68 et 14,44** ; Mistral-7B `top` 0,4764, `bottom` 0,4776 (10⁻³), `free` 0,6012, **G = 12,48 et 12,36**.
  La lecture s'arrête si une référence ne redonne pas sa valeur à 10⁻⁴ près.

## Test, écrit d'avance (le test U de l'isolation)
**U = dégelé (r = 1) − gelé (r = 2)**, apparié par graine, contre les G fixés : IC95 au-dessus de 0,7·G → le gel explique l'essentiel ;
au-dessous de 0,3·G → l'essentiel vient d'ailleurs ; sinon partagé. Pour chaque bande et chaque modèle.
Rapportés : les taux retenus, les scores, l'écart du bras dégelé au LoRA libre, le modèle de carte.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés.
