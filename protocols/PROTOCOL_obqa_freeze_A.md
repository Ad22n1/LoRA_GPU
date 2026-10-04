# Protocole pré-enregistré — geler A plutôt que B, sur OpenBookQA au rang 2

Écrit le 26/09 et committé AVANT tout run de cette campagne ; déposé publiquement (OSF) avant le lancement si possible — le lien et
l'heure du dépôt sont alors consignés dans `PROTOCOL_obqa_freeze_A_osf.md`.

## Question
Sur la tâche de format avec Llama-3.2-1B (`PROTOCOL_freeze_A.md`), geler A plutôt que B coûte beaucoup moins (+12,96 et +18,96
points, établis) et presque rien face au LoRA libre : le coût du gel est propre au facteur de sortie. Ce résultat tient-il sur un
benchmark public, avec un autre modèle ? OpenBookQA au rang 2, sur Qwen2.5-1.5B, où le gel de B coûte 8,04 et 12,36 points, et où
dégeler B récupère tout l'écart (`PROTOCOL_obqa_unfrozen.md`).

## Montage
Qwen2.5-1.5B, OpenBookQA (empreinte `dc05f4d847f9`), rang 2, base de la grille `obqa_r2_common`.
- **Bras** : `dual_top`, `dual_bottom` (A fixé sur les lignes de Vᵀ de la bande dominante ou mineure, B entraîné à partir de zéro)
  et `dual_random_ortho` (A sur une base orthonormée aléatoire, salée).
- **Références existantes, appariées par graine (2 à 6)** : `top` gelé à 5·10⁻³ (0,4652), `bottom` gelé à 2·10⁻³ (0,4220),
  `free` au rang 1 à 2·10⁻³ (0,5456). La lecture s'arrête si elle ne les retrouve pas, ou si elles ont des doublons.
- **Sélection et rapport en une nuit** : sept taux (10⁻⁴ à 2·10⁻²) sur les graines 0 à 6 ; taux choisi en validation sur les
  graines 0 et 1 ; seuls les runs des graines 2 à 6 à ce taux sont lus ; un taux au bord est rapporté comme une limite. Facteurs
  finaux sauvegardés, pour une réévaluation éventuelle sur un seul modèle de carte.
- **Budget, déclaré** : un bras qui gèle A entraîne r × d_out paramètres, un bras qui gèle B r × d_in. Sur Qwen2.5-1.5B, l'attention
  groupée rend k et v étroits en sortie (256, contre 1 536 en entrée) : l'écart n'est pas celui de Llama. Comptes exacts, 28 couches :
  **1 290 240** pour un bras qui gèle A au rang 2, contre **1 017 856** pour un bras qui gèle B (**+26,8 %**) et **1 154 048** pour
  `free` au rang 1 (**+11,8 %**) — ces deux derniers identiques aux comptes déjà enregistrés par les runs. Référence pour juger ce surplus : `top_unfrozen` et
  `bottom_unfrozen` au rang 1, qui ont exactement le budget de `free` (0,5524 et 0,5496).

- **Un seul modèle de carte** : les runs de cette campagne sont épinglés sur les machines RTX 4000 Ada, en excluant toutes les autres
  (y compris celles dont le modèle est inconnu) ; la liste est calculée au lancement à partir des runs enregistrés et consignée dans
  `~/lora-grid/obqa_dual/machines_epinglees.txt`. Les références gelées, plus anciennes, ont tourné sur des cartes mêlées : le modèle
  de carte de chaque run est rapporté, et les tests sont refaits sur les graines où tous les runs comparés partagent un modèle.

## Tests, écrits d'avance
1. **Principal, pour chaque bande X** : Δ_X = `dual_X` − X gelé, apparié sur les graines 2 à 6, **t > 3,495** (Bonferroni sur deux).
   Issue 1 : établi et positif aux deux bandes — le coût est propre au facteur de sortie sur les deux tâches ; issue 2 : à une seule ;
   issue 3 : à aucune, ou négatif.
2. **Pré-enregistré ici** (post hoc sur la tâche de format) : l'écart entre bandes quand A est gelé, contre celui quand B est gelé :
   (`dual_bottom` − `dual_top`) − (`bottom` − `top`), apparié, bilatéral, **t > 2,776**.
3. **Rapportés, sans issue** : les bras qui gèlent A contre `free` et contre les bras dégelés au rang 1 ; l'étendue entre les trois
   bras qui gèlent A ; le déplacement du taux ; le modèle de PARTITION de chaque run, et les tests (1) et (2) refaits sur les seules graines
   dont tous les runs comparés partagent un modèle de carte (non calculable sous trois graines).

## Engagements
**C'est la dernière campagne de la v1.** Après elle et la réévaluation sur un seul modèle de carte, aucune campagne
n'est lancée avant le gel du 30/09 : le temps va à l'écriture.
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
