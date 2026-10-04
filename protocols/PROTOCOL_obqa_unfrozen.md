# Protocole pré-enregistré — le coût du gel de B, isolé, sur OpenBookQA au rang 2

Écrit le 26/09 et committé AVANT tout run de cette campagne. S'il est aussi déposé publiquement (OSF) avant le lancement, le lien
et l'heure du dépôt sont consignés dans une note datée, `PROTOCOL_obqa_unfrozen_osf.md`.

## Question
Sur la tâche de format avec Llama-3.2-1B (`PROTOCOL_top_unfrozen.md` et son additif), dégeler B à rang et initialisation égaux
rapporte tout l'écart au LoRA libre, et l'écart entre les deux bandes rétrécit quand B est entraîné (5,93 points, test post hoc).
Ce résultat tient-il sur un benchmark public, avec un autre modèle ? OpenBookQA au rang 2, sur Qwen2.5-1.5B, où le gel coûte 8,04
points (établi, `PROTOCOL_obqa_rank2.md`).

## Montage
Exactement celui de l'expérience 1, transposé : Qwen2.5-1.5B, OpenBookQA (empreinte `dc05f4d847f9`), base de la grille `obqa_r2_common`.
- **Bras** : `top_unfrozen` et `bottom_unfrozen` (la base de la bande comme B initial, A = 0, B entraîné), aux rangs 1 et 2.
- **Références existantes, appariées par graine (2 à 6)** : `top` gelé à 5·10⁻³, `bottom` gelé à 2·10⁻³ (rang 2), `free` au rang 1 à
  2·10⁻³ — leurs taux retenus dans D.
- **Sélection et rapport en une nuit** : sept taux (2·10⁻⁴ à 2·10⁻²) sur les graines 0 à 6 ; le taux est le meilleur en validation
  (`val_mcq_acc`) sur les graines 0 et 1 ; seuls les runs des graines 2 à 6 à ce taux sont lus. Un taux retenu au bord est rapporté
  comme une limite. Les facteurs sont sauvegardés au pas 0 et en fin de run, pour les angles.
- **Budget** : `unfrozen` au rang 1 a le budget de `free` au rang 1, soit 13,4 % de plus que les bras gelés au rang 2 ; `unfrozen` au
  rang 2 a 2,27 fois leur budget. Déclaré.

## Valeurs fixées d'avance
G = `free` (r=1) − gelé (r=2) : **8,04 points pour `top`, 12,36 pour `bottom`** (graines 2 à 6). La lecture s'arrête si elle ne les
retrouve pas à 0,01 près : sinon U serait comparé à une autre référence.

## Tests et issues, écrits d'avance
1. **Principal, pour chaque bande** : U = `unfrozen` (r=1) − gelé (r=2), apparié par graine, avec son intervalle à 95 % (t, 4 d.l.).
   Le gel explique l'essentiel si **toute** l'intervalle est au-dessus de 0,7·G ; l'essentiel vient de l'initialisation, de la base
   ou du rang si toute l'intervalle est sous 0,3·G ; sinon : partagé, sans attribution.
2. **Le gel seul, pour chaque bande** : `unfrozen` (r=2) − gelé (r=2), t > 3,495 (Bonferroni sur les deux bandes).
3. **Le rétrécissement de l'écart entre bandes, pré-enregistré ici** (post hoc sur la tâche de format) : au rang 2,
   (`bottom` − `top`) dégelés moins (`bottom` − `top`) gelés, apparié par graine, bilatéral, **t > 2,776**.
4. **Rapportés, sans issue** : la décomposition G = [free − unfrozen, r=1] + [unfrozen r=1 − r=2] + [unfrozen r=2 − gelé r=2], avec un
   intervalle sur chaque terme ; l'écart entre bandes dégelées et son équivalence à ±1 point ; les angles principaux entre le B initial
   et le B final.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
