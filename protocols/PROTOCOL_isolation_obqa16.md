# Protocole pré-enregistré — le gel de B isolé au rang de MiCA, sur OpenBookQA (Qwen2.5-1.5B, rang 16)

Écrit et committé AVANT tout run ; sa date est celle de son commit. Série de la sixième révision de l'engagement.
Arrêt des lancements fixé au **lundi 28/09 à 20 h** : ce qui n'est pas parti à cette heure n'entre pas dans la v1.

## Pourquoi
Le bras `bottom` est la paramétrisation de MiCA (B = les r dernières colonnes de U, A = 0, B figé ; vérifié sur le code de PEFT 0.21.0).
Au rang 16, où MiCA opère, le gel de B n'est isolé que sur ARC-Challenge. Cette campagne l'isole sur une seconde tâche, OpenBookQA.

## Montage (celui de l'isolation au rang 16 sur ARC-Challenge)
- Qwen2.5-1.5B, OpenBookQA (empreinte `dc05f4d847f9`). **Base : la configuration des runs de référence eux-mêmes**, lue sur le serveur ;
  seuls changent le mode, le rang (α = r), le taux et la graine.
- **Nouveaux bras** : `top_unfrozen` et `bottom_unfrozen` (B entraîné depuis la même base et la même initialisation), **au rang 16** (celui
  des bras gelés) **et au rang 7** (celui du LoRA libre à budget égal) ; sept taux (10⁻⁴ à 10⁻²) ; **toutes les graines à tous les taux**
  (0 et 1 : sélection en validation ; 2 à 6 : rapport). 196 runs.
- **Références** : `top` gelé au rang 16 à 2·10⁻³, `bottom` gelé au rang 16 à 10⁻³, et le LoRA libre au rang 7 à son taux retenu (le seul
  taux où il a ses cinq graines de rapport ; le déploiement s'arrête s'il n'est pas unique). **Leurs valeurs graine par graine, et les G qui
  en découlent, sont calculés et committés au déploiement, avant tout run** (`refs_o16.json`).
- **Contrôle à carte égale** : les trois références sont réentraînées à l'identique sur RTX 4000 Ada, graines 2 à 6, dans `~/lora-runs-suivi`
  (15 runs). **211 runs**, tous épinglés sur RTX 4000 Ada.

## Tests, écrits d'avance (ceux de l'isolation)
1. **U = dégelé (rang 7) − gelé (rang 16)**, contre G = libre (rang 7) − gelé (rang 16) : IC95 au-dessus de 0,7·G → le gel explique
   l'essentiel ; au-dessous de 0,3·G → l'essentiel vient d'ailleurs ; sinon partagé. **Plancher : le test U ne s'applique que si G ≥ 2 points** ;
   sinon il est rapporté comme non applicable.
2. **Le gel seul** : dégelé (rang 16) − gelé (rang 16), pour chaque bande, apparié, **t > 3,495**.
3. **Le rétrécissement de l'écart entre bandes** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`), au rang 16, **t > 2,776**, lu
   comme un rétrécissement seulement si l'écart gelé vaut au moins 2 points en valeur absolue.
Chaque test est calculé au taux retenu (principal) et au taux oracle (borne optimiste), contre les références d'origine (principal) et contre
les références réentraînées sur la même carte (contrôle). Rapportés : les taux retenus, les modèles de carte de chaque série.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
