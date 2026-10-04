# Protocole pré-enregistré — d'où vient l'écart entre `top_lrscale` et `top_scalar` ? Une comparaison de trajectoires

Écrit et committé AVANT tout run de cette campagne ; sa date est celle de son commit. Série de la sixième révision de l'engagement.

## La question
Le papier rapporte −0,83 point [−1,64 ; −0,02] entre `top_lrscale` (l'échelle par module mise dans le taux d'A) et `top_scalar` (la même
échelle mise dans B), « source not identified ». Sous Adam, sans écrêtage et avec ε → 0, les deux suivent la même trajectoire : les ΔW sont
égaux à chaque pas. Trois causes peuvent rompre l'identité : l'**écrêtage global** (il voit des gradients d'A différents d'un facteur s),
l'**ε d'Adam**, et la **précision numérique** (bf16). Cette campagne les isole, une à une.

## Changement de code (committé avec ce protocole)
Un champ `adam_eps` (défaut 10⁻⁸, la valeur par défaut de torch), passé aux deux constructions d'`AdamW`. À sa valeur par défaut il reste
hors de l'identifiant des runs ; le déploiement recalcule l'identifiant de tous les runs existants et s'arrête si un seul change.

## Montage
- Llama-3.2-1B, tâche de format, rang 2 (base `configs/grids/sigma_residual_C2.yaml`, vérifiée comme la campagne d'origine), au taux de
  `top_scalar` (5·10⁻³), **graine neuve 200**. Facteurs A et B sauvegardés à chaque pas jusqu'au 16ᵉ, puis tous les 4 pas.
- **Cinq conditions**, chacune avec `top_scalar` et `top_lrscale` :
  C0 identité (écrêtage désactivé : max_grad_norm = 10⁶ ; ε = 10⁻²⁰ ; fp32) ; C1 = C0 + écrêtage à 1,0 ; C2 = C0 + ε = 10⁻⁸ ;
  C3 = C0 + bf16 ; C4 = réglage du papier (écrêtage 1,0, ε = 10⁻⁸, bf16).
- **Témoin** : `top_scalar` en C0, relancé une seconde fois (dossier séparé) : il mesure le plancher de non-déterminisme du matériel.
- **11 runs**, épinglés sur RTX 4000 Ada, dans `~/lora-runs-traj` et `~/lora-runs-traj-ctrl`, hors de la base principale.

## Mesure et règle, écrites d'avance
À chaque pas sauvegardé, D = ‖ΔW(`top_lrscale`) − ΔW(`top_scalar`)‖_F / ‖ΔW(`top_scalar`)‖_F, sommé sur tous les modules
(ΔW = échelle · B·A). Plancher = le plus grand D entre les deux runs du témoin, **et au moins 10⁻⁶** : l'arrondi en fp32 laisse à lui seul un écart
relatif de cet ordre entre deux bras qui stockent U et s·U, même si le matériel est parfaitement déterministe.
- **L'identité tient en C0** si D(C0) reste sous 10 × le plancher à tous les pas.
- **Un facteur est une cause** si, dans sa condition, le D final dépasse 10 × max(plancher, D final de C0).
- Si C4 diverge et qu'aucun facteur seul ne le fait, la cause est **une interaction** entre facteurs, rapportée comme telle.
Rapportés : D pas à pas, les scores finaux des deux bras dans chaque condition, et l'écart entre eux (une seule graine : descriptif).

## Ce que la campagne ne fait pas
Elle ne reteste pas l'écart de −0,83 : une seule graine ne le permet pas. Elle localise le mécanisme qui peut le produire. La phrase du papier
« its source is not identified » ne change que si une cause est nettement identifiée.
