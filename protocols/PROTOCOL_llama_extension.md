# Protocole pré-enregistré — extension du balayage β sur Llama-3.2-1B

**À committer avant tout lancement.** Rien de ce qui suit ne change après avoir vu un score.

## Pourquoi
Sur les graines 2 à 8, β=0,75 est la meilleure des cinq positions, mais seules deux comparaisons
sur quatre passent Bonferroni ; contre β=0,5 l'écart est de 0,98 point (t=0,99). Sept graines n'ont
pas la puissance pour un écart de cette taille. Qwen, à quinze graines, en passe trois sur quatre.

## Ce qui est fixé d'avance
- **Graines neuves : 9 à 16**, huit graines. Les graines 0 et 1 (sélection du taux) sont exclues de
  tout test. La graine 9 existe déjà pour β=0 à 0,75 à leur taux sélectionné : **aucun de ses scores n'a été
  regardé**. Pour β=1, la graine 9 n'existe qu'à 10⁻² (ancienne grille sans taux par position) : ce run
  n'est pas au taux sélectionné, il n'entre dans aucun test, et β=1 graine 9 est relancée à 5·10⁻³.
- **Taux : ceux de `configs/lr_selected.yaml` tels qu'au commit de ce fichier**
  (md5 `278ede112756a920182d7129a715203e`) : 10⁻² pour β=0 à 0,75, 5·10⁻³ pour β=1. `select_lr` n'est pas
  relancé sur le rang 2 avant la fin de l'extension.
- **Configuration identique** aux graines 2 à 8 : rang 2, 7 modules, écrêtage 1,0, 614 items,
  même vérificateur, même données.
- **Direction** : fixée par les graines 2 à 8 (β=0,75 au-dessus des quatre autres), donc test
  **unilatéral**, comme pour Qwen.

## Le test principal, le seul qui compte
Test t apparié par graine, β=0,75 contre chacune des quatre autres positions **et contre
`random_ortho`** (base orthonormale tirée hors du spectre, à 10⁻², sel activé), **sur les huit graines
neuves seules** : c'est une réplication indépendante de ce qui a été trouvé sur 2 à 8, où l'écart à
`random_ortho` était de +2,42 pt, t=2,94, non établi une fois compté dans la famille.
Correction de Bonferroni sur **cinq** : **t > 2,998** à 7 ddl. En complément, pour ne pas dépendre de la
normalité : test des signes et Wilcoxon apparié exact, rapportés à côté de chaque t.

## Rapporté en second, sans en faire le résultat
Les mêmes cinq tests sur les quinze graines 2 à 16 : seuil **t > 2,624** à 14 ddl.

## Engagements
- Les cinq comparaisons sont rapportées **quel que soit leur résultat**, dans le papier.
- Aucune graine n'est ajoutée après avoir vu un résultat. Si un run échoue (nœud, préemption), il
  est relancé à l'identique ; si une cellule reste manquante, la graine entière est exclue des
  cinq positions et on le dit.
- Si β=0,75 n'est plus la meilleure position sur les graines neuves, on l'écrit.
