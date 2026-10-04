# Protocole pré-enregistré — B gelé sur les directions du gradient (le placement sans oracle)

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : mardi 29/09 a 08 h 00 (2026-09-29 08:00).** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Pourquoi
La phase 2 montre que B gelé sur les directions d'un LoRA libre entraîné récupère l'essentiel de l'écart, mais ces directions supposent un
entraînement préalable sur la tâche. Ici, les directions sont calculées avant tout entraînement, à partir du gradient de la perte. LoRA-One
(Zhang et al., ICML 2025) prouve que les adaptateurs s'alignent sur des sous-espaces singuliers de ce gradient, et LoRA-SB (Ponkshe et al.,
2024) gèle B et A sur une approximation de la première mise à jour : ce protocole teste la même idée pour un B seul gelé, dans notre cadre,
sans revendiquer la méthode.

## Les directions
Pour chaque module, le gradient de la perte d'entraînement par rapport à W₀, moyenné sur 64 exemples du jeu d'entraînement (64 séquences
d'entraînement ; sur OpenBookQA, des blocs de 384 tokens ; tirées avec une graine fixe, 1000, qui ne sert à aucun score rapporté), au modèle
de base, sans aucune mise à jour. La base du nouveau bras, `grad`, est formée des 2 premiers vecteurs singuliers à gauche de ce gradient
(côté sortie), orthonormés. Les bases sont calculées une fois, sauvegardées et committées avant tout run.

## Montage
- Deux paires, rang 2 : OpenBookQA avec Qwen2.5-1.5B, et la tâche de format avec Llama-3.2-1B ; configuration reprise des runs de référence.
- Bras `grad` : B fixé sur cette base, sans Σ, gelé ; A entraîné depuis zéro ; même budget que `top` (r · d_in par module). Mode « B chargé
  depuis un fichier », déjà testé en phase 2.
- Taux : cinq (les cinq de la phase 2 : 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻²), sélection en validation sur les graines 0 et 1, rapport sur
  les graines 2 à 6, un cran de prolongement si l'optimum tombe au bord.
- Références publiées : `top`, `bottom`, `random_ortho`, LoRA libre au rang 1, et `learned` (phase 2 et réplique B).
- Une seule carte, RTX 4000 Ada. Environ 30 runs, 5 PARTITION-heures.

## Test et lecture, fixés d'avance (comme pour la phase 2)
Sur l'intervalle à 95 % de `grad` − `top`, apparié par graine, avec G fixé d'avance (8,04 sur OpenBookQA, 12,41 sur la tâche de format) :
- borne basse ≥ 0,7·G : les directions du gradient suffisent à placer B ; l'essentiel du coût se récupère sans oracle ;
- borne haute < 0,3·G, que le test soit établi ou non : les directions du gradient ne suffisent pas ; le placement utile demande un entraînement ;
- sinon : non résolu.
t > 2,776 rapporté à côté. Rapportés sans test : `grad` − `learned`, `grad` − `random_ortho`, `grad` − LoRA libre.

## Règles communes
Une relance pour échec machine ; rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés hors le cran de prolongement.
