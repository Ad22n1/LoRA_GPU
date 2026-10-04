# Protocole pré-enregistré — la comparaison principale au rang 2, sur graines neuves et un seul modèle de carte

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit. Il appartient à la série de la sixième révision de
l'engagement (`PROTOCOL_exp1_format_qwen.md`), décidée par l'auteur.

## Pourquoi
L'exemple qui ouvre l'introduction — deux bandes gelées à cinq points l'une de l'autre, qui finissent à moins d'un point une fois B entraîné —
repose sur un test **post hoc** (+5,93 points, t = 4,96), et « gap minus spread » au rang 2 aussi (+5,4). Onze des runs d'origine n'ont pas
de modèle de carte enregistré. Cette campagne refait la comparaison sur **des graines neuves, épinglées sur RTX 4000 Ada**, avec des tests
écrits d'avance.

## Montage
- Llama-3.2-1B, tâche de format, rang 2 (base : `configs/grids/sigma_residual_C2.yaml`). **Même configuration que la campagne d'origine** :
  le générateur s'arrête si la base diffère sur la tâche, le modèle, les 600 000 tokens, la longueur 384, le lot 2, l'accumulation 16, les sept
  modules ou l'**écrêtage à 1,0**.
- **Onze bras**, chacun **au taux du papier**, sans nouvelle sélection :
  `top`, `random`, `random_ortho` 10⁻² ; `bottom` **5·10⁻³** (celui du papier, et non le 2·10⁻³ de la sélection à cinq graines) (gelés, r = 2) ;
  `free` 10⁻³ (r = 1) ; `top_unfrozen` 5·10⁻³ (r = 1 et r = 2) ; `bottom_unfrozen` 5·10⁻³ (r = 1) et 2·10⁻³ (r = 2) ;
  `dual_top` et `dual_bottom` 10⁻² (A gelé, r = 2 ; taux du protocole du gel de A).
- **Graines neuves 120 à 124**, jamais utilisées. **55 runs**, épinglés sur RTX 4000 Ada, dans 11 grilles.
- Valeurs d'origine, pour référence (graines 2 à 6) : `top` 0,6443, `bottom` 0,5922, `free` 0,7684 ; dégelés au rang 2 0,8000 et 0,8072 ; au
  rang 1 0,7928 et 0,7954 ; `dual_top` 0,7739, `dual_bottom` 0,7818 ; G = 12,41 et 17,62.

## Tests, écrits d'avance
1. **PRINCIPAL — le rétrécissement de l'écart entre bandes quand B est entraîné** : (`bottom_unfrozen` − `top_unfrozen`) − (`bottom` − `top`),
   au rang 2, apparié par graine, bilatéral, **t > 2,776**. Plancher : l'écart gelé doit valoir au moins 2 points en valeur absolue pour que
   le test soit lu comme un rétrécissement.
2. **Le gel seul** : `X_unfrozen` (r = 2) − `X` (r = 2), pour X = `top`, `bottom`, **t > 3,495** (Bonferroni sur les deux bandes).
3. **U = dégelé (r = 1) − gelé (r = 2)**, contre **G recalculé sur les graines 120 à 124** (`free` r = 1 et bras gelés de cette campagne) :
   IC95 au-dessus de 0,7·G → le gel explique l'essentiel ; au-dessous de 0,3·G → l'essentiel vient d'ailleurs ; sinon partagé. Le G des
   graines 2 à 6 est rapporté à côté, pour référence.
4. **Le rétrécissement de l'écart entre bandes quand A est gelé à la place de B** : (`dual_bottom` − `dual_top`) − (`bottom` − `top`),
   apparié, bilatéral, **t > 2,776**, avec le même plancher — le test d'origine était post hoc ; il est ici pré-enregistré.
Rapportés sans test : **« gap minus spread »** (le gap au LoRA libre moins l'écart entre le meilleur et le pire des quatre bras gelés),
**descriptif** ; les scores de chaque bras et leur écart aux valeurs d'origine ; le modèle de carte de chaque run.

## Engagements
Rapporté quelle que soit l'issue. **L'introduction garde les chiffres d'origine et rapporte cette réplication, quelle que soit son issue** ;
si le rétrécissement n'est pas établi sur les graines neuves, elle le dit. Aucune graine ni aucun taux ajoutés.
