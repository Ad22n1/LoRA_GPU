# Protocole pré-enregistré — B gelé sur les directions d'un LoRA libre brièvement entraîné (préchauffage court)

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : jeudi 01/10 a 08 h 00 (2026-10-01 08:00).** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Question
La phase 2 montre qu'un B gelé sur les directions d'un LoRA libre entièrement entraîné récupère l'essentiel de l'écart (`learned`), et le
bras `grad` que les directions du gradient initial ne suffisent pas. Quelle fraction d'entraînement suffit pour placer B ?

## Paires
OpenBookQA avec Qwen2.5-1.5B, et la tâche de format avec Llama-3.2-1B, au rang 2, dans la configuration des runs de référence.

## Directions
Un LoRA libre au rang 2, graine de direction 0, au taux **2·10⁻³ sur les deux paires : le taux du run de direction de la phase 2**, pour
que le point k = 100 de la courbe soit ce run lui-même. *(Précision avant commit : sur OpenBookQA, le taux retenu du LoRA libre au rang 2
par la tendance est 5·10⁻³ ; on garde 2·10⁻³, celui de la phase 2. Sur la tâche de format, 2·10⁻³ est aussi le taux retenu.)* Ordonnancement
complet habituel (cosinus, préchauffe de 5 %), prévu pour tout le budget. Ses facteurs sont sauvegardés après k % des pas d'optimisation,
k = 5, 10 et 25, arrondis au pas supérieur, et au dernier pas :
- OpenBookQA (49 pas) : 3, 5 et 13 pas, et 49 ;
- tâche de format (274 pas) : 14, 28 et 69 pas, et 274.

Pour chaque module, la base orthonormale de l'espace des colonnes de son B (QR), comme en phase 2. Les bases sont committées, avec leur
empreinte SHA-256, avant tout run des bras. Contrôle de reproduction, rapporté : au dernier pas, le recouvrement avec les bases de la
phase 2, qui avait la même configuration.

## Bras
`warm5`, `warm10`, `warm25` : B fixé sur la base correspondante, sans Σ, gelé ; A entraîné depuis zéro pendant les pas restants, avec son
propre ordonnancement (cosinus, préchauffe de 5 %) : 46, 44 et 36 pas sur OpenBookQA ; 260, 246 et 205 pas sur la tâche de format. Le
nombre de tokens par pas étant le même, le total, préchauffage compris, égale exactement celui des autres bras. Contrôles avant tout run :
ceux du mode « B chargé depuis un fichier » (`<commit>`) ; le nombre de pas est fixé dans la configuration de chaque run.
Déclaré :
- pendant ses k %, le LoRA de préchauffage détient deux fois le budget d'un bras à B gelé ;
- la graine 0 sert aussi de graine de direction, comme en phase 2 ;
- une seule graine de direction.

## Taux et graines
Cinq taux : 5·10⁻⁴, 10⁻³, 2·10⁻³, 5·10⁻³, 10⁻². Sélection en validation sur les graines 0 et 1, rapport sur les graines 2 à 6. Un cran
de prolongement si l'optimum tombe au bord de la grille.

## Test principal et lecture, fixés d'avance
Par paire : `warm10` − `top`, apparié par graine, intervalle à 95 %. Références fixées d'avance : `top` à sa valeur publiée (0,4652 et
0,6443) et G fixé (8,04 et 12,41).
- Borne basse ≥ 0,7·G : un préchauffage court suffit à placer B.
- Borne haute < 0,3·G : il ne suffit pas.
- Sinon : non résolu.

t > 2,776 rapporté à côté. La conclusion générale « un préchauffage court suffit » n'est tirée que si les deux paires l'atteignent. Si des
références réentraînées sur la même carte existent, les résultats sont aussi rapportés contre elles, en le disant, sans changer la lecture.

## Rapportés sans test
- `warm5` − `top` et `warm25` − `top`.
- La courbe dose–réponse de k = 0 (`grad`) à k = 100 (`learned` : 0,5588 et 0,8081).
- Chaque bras − LoRA libre au rang 1.
- Les angles principaux entre les bases à k = 5, 10, 25 et celle de `learned`, par type de module.

## Règles communes
- Une seule carte (RTX 4000 Ada) pour tous les runs de la campagne.
- Une relance pour échec machine. Les runs abandonnés faute de mémoire libre sont relancés hors relance, au plus trois fois. Couvre-feu
  respecté : un run interrompu est repris la nuit suivante, sans compter comme relance.
- Rapporté quelle que soit l'issue. Aucune graine ni aucun taux ajoutés hors le cran de prolongement.
- Tout écart est consigné dans un addendum daté, committé avant le run qu'il concerne.
