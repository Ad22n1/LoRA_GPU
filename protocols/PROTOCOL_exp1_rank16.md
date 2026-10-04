# Protocole pré-enregistré — le gel de B isolé au rang 16, sur la tâche de format et sur ARC-Challenge

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit.

## Quatrième révision de l'engagement — contre sa propre formule
La troisième révision (`PROTOCOL_shift_5seeds.md`) écrivait : « Aucune autre révision ne sera faite : après cette campagne, aucune campagne
avant le gel du 30/09, quelle qu'en soit la raison. » Cette campagne la rompt, par décision de l'auteur, et ce protocole le dit tel quel.
Raison : c'est la suite directe du résultat central, qui n'est isolé qu'au rang 2 ; elle dit si la baisse du coût du gel avec le rang vient
du gel lui-même, et, sur ARC-Challenge, si l'écart entre bandes rétrécit quand B est entraîné. Environ 400 runs, avant le gel.

## Montage
Comme l'expérience 1 (`PROTOCOL_top_unfrozen.md` et `PROTOCOL_obqa_unfrozen.md`), au rang 16.
- **Tâche de format, Llama-3.2-1B** (base : `configs/grids/sigma_residual_C2.yaml`, vérifiée). Références existantes, graines 2 à 6 :
  `top` gelé r=16 à 2·10⁻³ (**0,7756**), `bottom` gelé r=16 à 5·10⁻⁴ (**0,7554**), `free` r=8 à 2·10⁻³ (**0,8205**).
- **ARC-Challenge, Qwen2.5-1.5B** (base : `configs/grids/arcc_own_top.yaml`, vérifiée : empreinte `0755bcdbdce3`). Références existantes,
  graines 2 à 6 : `top` gelé r=16 à 5·10⁻⁴ (**0,4970**), `bottom` gelé r=16 à 10⁻³ (**0,4452**), `free` r=7 à 5·10⁻⁴ (**0,4952**).
- **Nouveaux bras** : `top_unfrozen` et `bottom_unfrozen` (même base que la bande gelée, B entraîné), **au rang du LoRA libre de référence**
  (8 pour la tâche de format, **7 pour ARC**, comme l'expérience 1 comparait le rang 1 au rang 1) **et au rang 16**.
- Sept taux (format : 10⁻⁴ à 10⁻² ; ARC : 5·10⁻⁵ à 5·10⁻³), graines 0 à 6 ; taux choisi en validation sur 0 et 1 ; seuls les runs 2 à 6
  à ce taux sont lus ; un taux au bord est une limite. Épinglé sur les RTX 4000 Ada. 4 grilles, **392 runs**.
- **G fixés d'avance** (points, `free` − bande gelée) : format **4,49** (`top`) et **6,51** (`bottom`) ; ARC **−0,18** (`top`) et **5,00** (`bottom`).
  La lecture s'arrête si une référence ne redonne pas sa valeur à 10⁻⁴ près, ou a des doublons.

## Tests, écrits d'avance (ceux de l'expérience 1), à chaque tâche
1. **U = dégelé (rang du LoRA libre) − gelé (rang 16)**, contre G : IC95 entièrement au-dessus de 0,7·G → le gel explique l'essentiel ;
   entièrement au-dessous de 0,3·G → l'essentiel vient d'ailleurs ; sinon partagé. **Seulement si G ≥ 2 points** (format : les deux bandes ;
   ARC : `bottom`) ; pour `top` sur ARC (G = −0,18), U est rapporté sans issue.
2. **Le gel seul** : dégelé (16) − gelé (16), apparié, **t > 3,495** (Bonferroni sur les deux bandes).
3. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), au rang 16, apparié,
   bilatéral, **t > 2,776**. Sur ARC, c'est le test principal : l'écart gelé y vaut −5,18 points.
Rapportés : les taux retenus ; le modèle de carte de chaque run ; les tests (1)-(3) refaits sur les graines dont tous les runs comparés
partagent un modèle de carte (non calculable sous trois graines).

## Étape prévue si le contrôle à carte égale est incalculable
Les références gelées ont été évaluées sur des cartes variées ; les nouveaux runs sont tous sur RTX 4000 Ada. **Si le contrôle à carte égale du
test (3) porte sur moins de trois graines pour une tâche**, les quatre références gelées (`top` et `bottom` au rang 16, à leur taux retenu,
graines 2 à 6, des deux tâches) sont **réentraînées, épinglées sur RTX 4000 Ada**, dans un dossier séparé (`~/lora-runs-suivi`, hors
population principale), et les tests (1) à (3) sont recalculés avec elles : toutes les évaluations comparées sont alors faites, dans le
processus d'entraînement, sur un seul modèle de carte. Les grilles de cette étape (`configs/grids/e1r16_ref_*.yaml`, 20 runs) sont committées
avec ce protocole ; la lecture annonce elle-même si l'étape est déclenchée. Les résultats sont rapportés à côté de ceux de la campagne, et le
verdict pré-enregistré reste celui de la campagne. Ce n'est pas un suivi décidé après coup : c'est une étape de ce protocole.

## Encadrement par le taux oracle (contrôle de robustesse)
Toutes les graines tournent à tous les taux : chaque test (1) à (3) est aussi calculé avec les bras dégelés à leur taux **oracle** — le meilleur
en test sur les graines rapportées 2 à 6 —, les références restant à leur taux dans les deux colonnes. Optimiste par construction, cet
encadrement **ne remplace pas** le verdict au taux retenu ; il dit si ce verdict tient au bruit de la sélection sur deux graines, que les
courbes plus plates du rang 16 rendent plus fragile, surtout sur ARC.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
