# Protocole pré-enregistré — le gel de B isolé sur la tâche de format avec Qwen2.5-1.5B, au rang 2

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit.

## Sixième révision de l'engagement
L'additif de la campagne d'échelle (`PROTOCOL_scale.md`) écrivait qu'elle « n'ouvre aucune autre révision ». Cette campagne la rompt, par
décision de l'auteur, avec deux autres de la même série (réévaluation sur une carte, trajectoire du taux par module), toutes écrites
avant leurs runs. Raison : c'est la faiblesse W7 d'un relecteur — la tâche et le modèle ne sont pas encore séparés dans le résultat central.

## Montage (celui de l'expérience 1, sur un second modèle)
- Qwen2.5-1.5B, tâche de format (base : `configs/grids/scale_qwen15.yaml`, vérifiée), sept modules, écrêtage 1,0, α = r.
- **Références existantes**, campagne pré-enregistrée `PROTOCOL_format_qwen_rank.md`, graines 22 à 26, épinglées sur RTX 4000 Ada :
  `top` gelé r=2 à 10⁻² (**0,6257**), `bottom` gelé r=2 à 5·10⁻³ (**0,6391**), `free` r=1 à 5·10⁻³ (**0,7879**).
- **Nouveaux bras** : `top_unfrozen` et `bottom_unfrozen` (même base que la bande gelée, B entraîné), **au rang 1** (celui du LoRA libre
  de référence) **et au rang 2**. Sept taux (10⁻⁴ à 2·10⁻², ceux de la campagne) ; graines 20 et 21 (sélection, en validation), 22 à 26
  (rapport, au seul taux retenu). Épinglés sur RTX 4000 Ada. 4 grilles, **196 runs**.
- **G fixés d'avance** (points, `free` − bande gelée) : **16,22** (`top`) et **14,88** (`bottom`). La lecture s'arrête si une référence ne
  redonne pas sa valeur à 10⁻⁴ près.

## Tests, écrits d'avance (ceux de l'expérience 1)
1. **U = dégelé (rang 1) − gelé (rang 2)**, contre G : IC95 au-dessus de 0,7·G → le gel explique l'essentiel ; au-dessous de 0,3·G →
   l'essentiel vient d'ailleurs ; sinon partagé.
2. **Le gel seul** : dégelé (rang 2) − gelé (rang 2), apparié, **t > 3,495** (Bonferroni sur les deux bandes).
3. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), au rang 2, **t > 2,776**.
   Plancher déclaré : l'écart gelé vaut +1,34 point (`bottom` devant) ; s'il est inférieur à 2 points en valeur absolue, le test (3) est
   rapporté mais n'est pas lu comme un rétrécissement, faute d'écart à rétrécir.
Rapportés : les taux retenus, le modèle de carte de chaque run, les tests à carte égale, et l'encadrement au taux oracle (borne optimiste).

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
