# Protocole pré-enregistré — B gelé sur le gradient initial accumulé

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : jeudi 01/10 a 08 h 00 (2026-10-01 08:00).** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Question
Le bras `grad` (B gelé sur les 2 premiers vecteurs singuliers à gauche du gradient initial, moyenné sur 64 séquences ; commit `<commit>`)
ne récupère pas l'écart à LoRA. Cet échec vient-il d'une estimation trop pauvre du gradient ?

## Paires
OpenBookQA avec Qwen2.5-1.5B, et la tâche de format avec Llama-3.2-1B, au rang 2, dans la configuration des runs de référence (600 000
tokens supervisés, α = r, sept modules, écrêtage à 1,0).

## Bases
Pour chaque module, le gradient de la perte d'entraînement par rapport à W₀, au modèle de base et sans mise à jour, est moyenné sur :
- 256 séquences (`grad256`) ;
- 1 024 séquences (`grad1024`), **sur la tâche de format seulement** : le jeu d'entraînement d'OpenBookQA ne compte que 272 séquences ;
- tout le jeu d'entraînement (`gradall` : 272 séquences sur OpenBookQA, 3 178 sur la tâche de format).

Les ensembles sont emboîtés par construction : d'abord les 64 séquences de `grad` (le même tirage, graine 1000, commit `<commit>`), puis les
autres séquences dans une permutation fixe (graine 1000) ; avec 64 séquences, on retrouve exactement l'ensemble et les bases de `grad`.
*Correction avant commit : le texte initial disait « les 64 premières d'une même permutation » ; avec la fonction de tirage de Python,
ce n'était vrai que sur OpenBookQA.*
La perte est celle de l'entraînement : question et réponse sur OpenBookQA, réponse seule sur la tâche de format. B = les 2 premiers
vecteurs singuliers à gauche, SVD en double précision, sans Σ. Les bases sont calculées et committées, avec leur empreinte SHA-256, avant
tout run d'entraînement.

## Bras
`grad256`, `grad1024` (tâche de format seulement), `gradall` : B fixé sur la base correspondante et gelé ; A entraîné depuis zéro sur tout
le budget ; même nombre de paramètres entraînés que `top` (r·d_in par module). Contrôles avant tout run, committés : B égal au fichier, B
inchangé pendant l'entraînement, ΔW = 0 à l'initialisation, nombre de paramètres exact, identifiants des autres modes inchangés (le mode
« B chargé depuis un fichier » et son test, `<commit>`).

## Taux et graines
Cinq taux : 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻². Sélection en validation sur les graines 0 et 1, rapport sur les graines 2 à 6. Un cran
de prolongement si l'optimum tombe au bord de la grille.

## Test principal et lecture, fixés d'avance
Par paire : `gradall` − `top`, apparié par graine, intervalle à 95 %. Références fixées d'avance : `top` à sa valeur publiée (0,4652 sur
OpenBookQA, 0,6443 sur la tâche de format) et G fixé (8,04 et 12,41).
- Borne basse ≥ 0,7·G : le gradient accumulé suffit à placer B.
- Borne haute < 0,3·G : il ne suffit pas.
- Sinon : non résolu.

t > 2,776 rapporté à côté. La conclusion générale « le gradient accumulé suffit » n'est tirée que si les deux paires l'atteignent. Si des
références réentraînées sur la même carte existent (phase 2), les résultats sont aussi rapportés contre elles, en le disant, sans changer
la lecture.

## Rapportés sans test
- `grad256` − `top` et `grad1024` − `top` : la courbe de 64 séquences (`grad`) à tout le jeu.
- Chaque bras − `random_ortho` et − LoRA libre au rang 1.
- Les angles principaux entre les bases de `grad`, `grad256`, `grad1024`, `gradall` et celle de `learned` (graine de direction 0), par
  type de module.

## Règles communes
- Une seule carte (RTX 4000 Ada) pour tous les runs de la campagne.
- Une relance pour échec machine. Les runs abandonnés faute de mémoire libre sont relancés hors relance, au plus trois fois. Couvre-feu
  respecté : un run interrompu est repris la nuit suivante, sans compter comme relance.
- Rapporté quelle que soit l'issue. Aucune graine ni aucun taux ajoutés hors le cran de prolongement.
- Tout écart est consigné dans un addendum daté, committé avant le run qu'il concerne.
