# Protocole pré-enregistré — geler A plutôt que B

Écrit le 26/09 au matin et committé AVANT tout run de cette campagne. S'il est aussi déposé publiquement (OSF) avant le
lancement, le lien et l'heure du dépôt sont consignés dans une note datée, `PROTOCOL_freeze_A_osf.md`.

## Question
L'expérience `top_unfrozen` / `bottom_unfrozen` (protocole du 22/09 et son additif) montre qu'au rang 2, à rang et initialisation
égaux, dégeler B rapporte +15,57 points pour `top` et +21,50 pour `bottom` : le gel de B explique tout l'écart au LoRA libre.
Est-ce propre au gel de B, le facteur de SORTIE, ou geler n'importe quel facteur coûte-t-il autant ? Zhu et al. (asymétrie des
adaptateurs de bas rang) prédisent que geler A, le facteur d'ENTRÉE, coûte beaucoup moins.

## Montage
Llama-3.2-1B, tâche de format, rang 2, sept modules, écrêtage 1,0, α = r, 600 000 tokens — la base de la grille `tu_r2_all`.
- **Bras qui gèlent A** : `dual_top` et `dual_bottom` (A fixé sur les lignes de Vᵀ de la bande dominante ou mineure, B entraîné à
  partir de zéro) ; `dual_random_ortho` (A sur une base aléatoire orthonormée, salée), pour mesurer le coût du choix dans cette famille.
- **Budget, déclaré** : un bras qui gèle A entraîne r × d_out paramètres, contre r × d_in pour un bras qui gèle B. Sur Llama, par
  couche, 47 104 contre 40 960 au rang 2 : **+15 %**. Le `free` au rang 1 (44 032) en a 6,5 % de moins que les bras qui gèlent A.
- **Sélection et rapport en une nuit** : sept taux (10⁻⁴ à 2·10⁻²) sur les graines 0 à 6 ; le taux est le meilleur en validation sur
  les graines 0 et 1 ; seuls les runs des graines 2 à 6 à ce taux sont lus. Un taux au bord de la grille est rapporté comme une limite.
- **Références existantes, appariées par graine (2 à 6)** : `top` à 10⁻² et `bottom` à 5·10⁻³ (gel de B, écrêtage 1,0), `free`
  au rang 1 à 10⁻³ — les cellules de G = 12,41 ; la lecture s'arrête si elles contiennent des doublons.

## Test principal, issues écrites d'avance
Pour chaque extrémité X (`top`, `bottom`) : Δ_X = `dual_X` − X, apparié sur les graines 2 à 6, t bilatéral, **t > 3,495**
(Bonferroni sur les deux extrémités). Prédiction (Zhu et al.) : Δ_X > 0.
1. Δ établi et positif aux deux extrémités → geler A coûte moins que geler B : le coût est propre au facteur de sortie, et la
   contribution s'écrit « geler B coûte plus que choisir » ;
2. établi à une seule → rapporté extrémité par extrémité ;
3. établi à aucune, ou négatif → geler A coûte autant (ou plus) : le coût tient au gel d'un facteur, quel qu'il soit.
Réserve, quelle que soit l'issue : les bras qui gèlent A ont 15 % de budget en plus ; une issue 1 n'est donc pas un effet de budget
seulement si Δ dépasse nettement ce qu'un tel écart de budget peut expliquer — on le rapportera sans le trancher.

## Analyses secondaires, descriptives
L'écart au LoRA libre de la famille qui gèle A (`free` r=1 − meilleur bras `dual`) ; l'étendue entre bras `dual` (le coût du choix
dans cette famille) ; le déplacement du taux entre `dual_top` et `dual_bottom`.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés, hors un prolongement d'un cran si l'optimum tombe au bord.
