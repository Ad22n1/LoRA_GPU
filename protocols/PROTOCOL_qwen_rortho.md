# Protocole pré-enregistré — β=0,75 contre une base orthonormale tirée au hasard, sur Qwen

Écrit et committé **avant** le lancement, le 21/09 au soir.

## Pourquoi
Sur Llama, β=0,75 devance `random_ortho` sans que l'écart soit établi au test principal (t = 2,75 pour
2,998 sur huit graines neuves). Sur Qwen, la comparaison n'a jamais été faite : `random_ortho` n'existe
que sur les graines 2 à 6, alors que le balayage en β couvre les graines 2 à 16.

## Ce qui est lancé
`random_ortho` sur Qwen-2.5-0.5B, rang 2, graines **7 à 16** (dix graines), à 1e-2, son taux sélectionné
sur Qwen, tirage salé. β=0,75 existe déjà sur ces graines, à 1e-2, son taux sélectionné.
**Les scores de β=0,75 sur ces graines sont déjà connus** (moyenne 0,517, analyse Qwen du 21/09) ; ceux de
`random_ortho` ne le sont pas. Ce que ce protocole fixe avant de les voir, c'est la règle de décision
ci-dessous et le nombre de graines.

## Le test principal, le seul qui compte
Une seule comparaison, fixée d'avance : β=0,75 contre `random_ortho`, **sur les dix graines 7 à 16
seules**, test t apparié **unilatéral** dans le sens observé sur Llama (β=0,75 au-dessus).
Seuil : **t > 1,833** (9 ddl, α = 0,05, une seule comparaison, donc pas de Bonferroni).
À côté du t : le test des signes et le Wilcoxon apparié exact, unilatéraux.

## En second, sans revendication
Les mêmes tests sur les quinze graines 2 à 16 : seuil t > 1,761 (14 ddl).

## Engagements
- Le résultat est rapporté **quel qu'il soit**. S'il ne passe pas, le papier dira que le contrôle
  contre une base aléatoire n'est établi sur aucun des deux modèles.
- Aucune graine ne sera ajoutée après celles-ci pour cette comparaison.
- Les runs qui échoueraient sont relancés à l'identique ; une graine incomplète est exclue du test.
