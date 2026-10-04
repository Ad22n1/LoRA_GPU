# Additif à PROTOCOL_free_writes.md — règle de lecture et profil spectral (28/09)

Committé AVANT l'analyse d'OpenBookQA (Qwen2.5-1.5B), dont les facteurs viennent d'être produits et n'ont pas été regardés.
Sur la tâche de format (Llama-3.2-1B), l'analyse a déjà tourné le 28/09 : la règle y est appliquée **a posteriori**, et déclarée comme telle.

## Règle de lecture
Pour une direction (la colonne de B contre les vecteurs singuliers à gauche U de W₀ ; la ligne de A contre les vecteurs singuliers à droite V),
avec k = 2, la médiane sur les couches par type de module, puis la moyenne sur les graines 2 à 6 :
- **hors des bandes testées** si sa part d'énergie reste **sous 5 fois la référence nulle k/d**, à la fois dans la bande dominante et dans la
  bande mineure ;
- sinon, dans la bande où ce rapport est le plus grand : **dans cette bande** si sa part d'énergie y atteint **au moins 0,10** (10 % de
  l'énergie de la direction) ; sinon, **au-dessus du hasard mais diffuse** — plus proche de cette bande qu'une direction aléatoire, sans s'y
  concentrer.

**Ce que le papier en dira** : il ne parlera d'un mécanisme que si la direction de B est déclarée hors des bandes sur OpenBookQA (lecture
pré-enregistrée), et il le rapportera comme post hoc sur la tâche de format.

## Ajouts, descriptifs
- **Profil par dixième du spectre** : la part de l'énergie de la direction sur chaque dixième des vecteurs singuliers, du plus dominant au plus
  mineur, et la part hors de l'espace de W₀ (modules plus larges en sortie qu'en rang), contre la part attendue au hasard.
- **Médiane sur les couches**, en plus de la moyenne déjà rapportée.
- **Exactitude des runs analysés**, comparée à celle des runs d'origine (0,5456 sur OpenBookQA ; 0,7684 sur la tâche de format).

## Phase 2 : geler B sur la direction apprise par le LoRA libre

### Déclencheur
La phase 2 part seulement si, sur OpenBookQA (lecture pré-enregistrée), la direction de B est déclarée « hors des bandes testées ». Si elle est
déclarée « dans une bande » ou « au-dessus du hasard mais diffuse », la phase 2 ne part pas, et la raison est rapportée. La lecture post hoc de
la tâche de format ne déclenche rien.

### Montage (OpenBookQA, Qwen2.5-1.5B, rang 2)
- La direction : un run du LoRA libre au rang 2, graine 0, au taux 2·10⁻³, facteurs sauvegardés. Pour chaque module, la base du nouveau bras
  est la base orthonormale de l'espace des colonnes de son B (d_out × 2), obtenue par QR. La graine 0 ne sert à aucun score rapporté ; seules
  ses données d'entraînement définissent la direction.
- Le nouveau bras, `learned` : B fixé sur cette base, sans Σ, gelé ; A entraîné depuis zéro. Même budget que `top` et `bottom` au rang 2
  (r · d_in par module).
- Avant tout run, un test du nouveau mode : B chargé est identique au fichier à la précision près, B ne bouge pas pendant l'entraînement, et
  ΔW = 0 à l'initialisation.
- Taux : cinq (10⁻³, 2·10⁻³, 5·10⁻³, 10⁻², 2·10⁻²), sélection en validation sur les graines 0 et 1, rapport sur les graines 2 à 6 au taux
  retenu, un cran de prolongement si l'optimum tombe au bord.
- Références existantes, graines 2 à 6 : `top`, `bottom`, `random_ortho` (B gelé, r = 2), `free` (r = 1). Si une référence n'a pas tourné
  entièrement sur RTX 4000 Ada, elle est réentraînée à l'identique sur ce modèle.
- Une seule carte, RTX 4000 Ada. Environ 16 runs, 3 PARTITION-heures.

### Test, écrit d'avance
`learned` moins `top`, apparié par graine, t > 2,776 (un seul test).
Rapportés sans test : `learned` moins `bottom` et moins `random_ortho` ; la part de l'écart récupérée, P = (`learned` − `top`) / G, avec
G = `free` − `top` = 8,04 points, fixé d'avance comme pour U, et son intervalle. Le test t > 2,776 est rapporté à côté de la lecture.

### Lecture fixée d'avance (sur l'intervalle à 95 % de `learned` − `top`, comme pour U)
- Si sa borne basse atteint 0,7·G = 5,63 points : l'essentiel du coût du gel de B vient de l'endroit où B est gelé ; les bandes spectrales de
  W₀ ne contiennent pas la direction dont l'adaptateur a besoin, et un B gelé au bon endroit s'en passe.
- Si sa borne haute reste sous 0,3·G = 2,41 points, que le test soit établi ou non : l'endroit compte peu ; l'essentiel du coût vient du gel
  lui-même, c'est-à-dire de l'impossibilité d'ajuster B pendant l'entraînement.
- Sinon : non résolu (le terme du papier), rapporté comme tel.
(Deux corrections apportées au texte du relecteur avant tout commit : la lecture porte sur l'intervalle et non sur la valeur de P, comme pour
U ; et le cas « l'endroit compte peu » ne demande pas que le test soit établi.)

### Échéance
Les lancements de la phase 2 obéissent à l'arrêt de la v2 (mardi 29/09, 20 h). Si le déclencheur, le test du nouveau mode ou les runs ne
tiennent pas avant cette heure, la phase 2 est reportée à un protocole à part, avec ce même texte, et le papier rapporte la phase 1 seule.

## Ce qui ne change pas
Les mesures déjà définies (parts d'énergie à k = 2 et k = 16, base aléatoire, angles principaux), le taux (le taux publié, 10⁻³ sur la tâche
de format et 2·10⁻³ sur OpenBookQA, où les cinq graines ont leurs facteurs) et l'autotest préalable.
