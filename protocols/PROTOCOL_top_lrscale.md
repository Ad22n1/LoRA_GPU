# Protocole pré-enregistré — le scalaire de `top_scalar`, déplacé dans le taux d'apprentissage

Écrit le 26/09 et committé AVANT tout run de cette campagne.

## Révision de l'engagement du 26/09
L'engagement écrit dans `PROTOCOL_obqa_freeze_A.md` et `ETAT_EXPERIENCES.md` faisait de « geler A » sur OpenBookQA la dernière
campagne de la v1. Il est révisé une fois, pour cette seule campagne, parce qu'elle répond à une question que le papier laisse
ouverte — la lecture de Σ comme un taux par module est aujourd'hui dérivée, pas démontrée par un bras dédié —, pour 40 runs sur une
nuit. **Après elle, aucune campagne avant le gel du 30/09.**

## Question
Sous Adam, geler B = s·U_r (`top_scalar`, s = √(moyenne des σ de la bande), par module) équivaut à geler B = U_r (`top`) avec le taux
de A multiplié par s — à deux exceptions près : l'ε d'Adam et l'écrêtage GLOBAL du gradient, qui ne voient pas les deux bras de la même
façon. Dans notre configuration, ces deux détails cassent-ils l'équivalence ?

## Montage
- **Nouveau bras `top_lrscale`** : B = U_r (comme `top`), A entraîné, un groupe d'optimisation par module au taux lr × s, avec le MÊME s
  que `top_scalar` (calculé par la même ligne du code). Un test (`test_top_lrscale_follows_top_scalar_under_adam`) vérifie sur un
  petit modèle, sans écrêtage et avec ε → 0, que les deux bras suivent la même trajectoire (même ΔW à chaque pas).
- **Configuration identique à C2** (`PROTOCOL_sigma_residual_C2.md`) : Llama-3.2-1B, tâche de format, rang 2, sept modules, écrêtage 1,0,
  α = r, taux 5·10⁻³ (celui de `top_scalar`), **graines 65 à 104**, les 40 de C2.
- **Référence** : les runs de `top_scalar` de C2, sur les mêmes graines. **Leurs scores sont déjà connus** (C2) ; ceux de
  `top_lrscale` ne le sont pas. Ce protocole fixe la règle de décision avant de les voir.
- **Une seule comparaison** : l'identité ne dépend ni du taux, ni du rang, ni du modèle ; un autre choix ne changerait pas la réponse.

- **Un seul modèle de carte** : `top_lrscale` est épinglé sur les machines RTX 4000 Ada (toutes les autres, et celles dont le modèle
  est inconnu, sont exclues ; liste consignée dans `~/lora-grid/top_lrscale_C2/machines_epinglees.txt`). Les runs de `top_scalar`
  de C2 ont tourné sur des cartes mêlées : la répartition des cartes est rapportée pour les deux bras.
- **L'écrêtage, enregistré** : pour `top_lrscale`, la fraction des pas écrêtés et la norme maximale du gradient avant écrêtage (champs
  `clip_frac` et `grad_norm_pre_clip_max`, ajoutés au journal de l'entraînement le 26/09, sans changer l'écrêtage) ; pour
  `top_scalar`, seule sa norme moyenne avant écrêtage existe (`grad_norm_pre_clip_mean`). Dans `top_scalar`, le gradient de A est
  multiplié par le scalaire du module ; dans `top_lrscale`, il ne l'est pas : l'écrêtage global ne se déclenche pas aux mêmes pas.

## Test, écrit d'avance (celui de C2)
d = `top_lrscale` − `top_scalar`, apparié par graine sur les 40 graines, en points.
- **Équivalence** : l'IC90 de d entièrement dans **[−1 ; +1]** point → l'équivalence tient dans notre configuration : la lecture de Σ
  comme un taux par module est établie par l'expérience, écrêtage et ε compris.
- **Différence** : t > **2,023** (bilatéral, 39 d.l.) → l'écrêtage ou ε cassent l'équivalence ; le papier le dira.
- **Ni l'un ni l'autre : l'équivalence est « non montrée », PAS « l'identité est rompue ».** Les runs de `top_scalar` datent de C2 et
  ceux de `top_lrscale` tournent maintenant, peut-être sur d'autres cartes : la différence appariée contient aussi le bruit de relance
  (un à deux points par graine, mesuré le 26/09 sur OpenBookQA). Seule une différence établie (t > 2,023) serait lue comme une rupture,
  et l'écrêtage enregistré dira alors si elle en vient.
Puissance (calcul de C2, écart type de d ≈ 1,94 point) : 89 % d'établir l'équivalence à ±1 point avec 40 graines si les bras sont identiques.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ajoutée. Le modèle de PARTITION de chaque run est rapporté.
