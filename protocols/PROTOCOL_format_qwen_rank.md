# Protocole pré-enregistré — la tâche de format sur Qwen2.5-1.5B, aux rangs 2 et 16 : le modèle ou la tâche ?

Écrit le 26/09 et committé AVANT tout run de cette campagne.

## Seconde révision de l'engagement du 26/09
Le protocole du taux par module (`PROTOCOL_top_lrscale.md`) écrivait : « Après elle, aucune campagne avant le gel du 30/09 ». Cet engagement
est révisé une seconde fois, pour cette seule campagne, parce qu'elle répond à une question que le résumé du papier laisse ouverte, et
qu'elle se termine le jour même (392 runs, environ deux heures et demie en parallèle). **Après elle, aucune campagne avant le gel.**

## Question
Aujourd'hui, la tâche de format tourne sur Llama-3.2-1B et ARC sur Qwen2.5-1.5B : le modèle et la tâche sont confondus. Sur la tâche de
format avec Llama, geler coûte plus que choisir aux rangs 2 à 8 et l'ordre s'efface au rang 16 ; sur ARC avec Qwen, il s'inverse aux
rangs 16 et 32 et n'est pas établi au rang 2. La même tâche de format, sur Qwen2.5-1.5B, aux rangs 2 et 16, dit ce qui vient du modèle
et ce qui vient de la tâche. (Le rang contre la tâche est déjà partiellement tranché à l'intérieur d'ARC.)

## Montage
- Qwen2.5-1.5B, tâche de format (base : `configs/grids/scale_qwen15.yaml` du serveur, vérifiée : tâche de format et Qwen2.5-1.5B,
  sinon arrêt), sept modules, écrêtage 1,0, α = r, sel activé pour `random_ortho`.
- **Bras** : `top`, `bottom`, `random_ortho` au rang r, et `free` à budget égal (rang 1 pour r = 2, rang 7 pour r = 16), comme sur ARC.
- **Graines neuves** : sélection sur 20 et 21, rapport sur 22 à 26. Elles évitent que le lanceur réutilise, comme « déjà faits », des runs
  antérieurs de même configuration dont les valeurs ont été vues (leçon de l'audit du 26/09).
- **Taux** : rang 2 : 10⁻⁴ à 2·10⁻² (sept valeurs) ; rang 16 : 10⁻⁴ à 10⁻² (sept valeurs). Taux choisi en validation sur 20-21 ; seuls
  les runs 22-26 à ce taux sont lus ; un taux au bord est rapporté comme une limite.
- **Un seul modèle de carte** : épinglé sur les RTX 4000 Ada (liste consignée dans chaque dossier de grille).
- 4 grilles, **392 runs** : 3 bras × 7 taux × 7 graines + `free` 7 × 7, à chaque rang.

## Test, écrit d'avance (celui d'ARC)
À chaque rang : geler = `free` − meilleur bras gelé ; choisir = étendue entre les trois bras gelés ; **geler − choisir**, IC95 par
bootstrap apparié sur les graines 22-26 (20 000 tirages).
- **Rang 2** : IC entièrement > 0 → le résultat de Llama tient sur Qwen pour la même tâche ; entièrement < 0 → choisir coûte plus ;
  sinon non établi.
- **Rang 16** : IC entièrement < 0 (comme ARC) → **le renversement suit le MODÈLE** ; IC contenant 0 (comme la tâche de format sur Llama)
  → **il suit la TÂCHE** ; IC entièrement > 0 → geler coûte encore plus au rang 16 sur ce modèle et cette tâche.
Rapportés sans issue : les taux retenus, le déplacement du taux de `bottom`, le meilleur bras spectral contre `random_ortho`.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
