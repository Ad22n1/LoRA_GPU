# Protocole pré-enregistré — le reste entre `top_sigma` et `top_scalar`

Écrit et committé avant le premier run (24/09).

## Pourquoi
Sous Adam, multiplier un facteur figé par une constante revient à multiplier le taux du module (à ε et à l'écrêtage près).
`top_scalar` garde donc l'échelle MOYENNE de Σ par module ; `top_sigma` garde en plus sa RÉPARTITION entre les directions.
Sur cinq graines, `top_scalar` récupère 64 % du gain de `top_sigma`, et le reste, +1,30 point (t = 2,21), n'est ni établi ni exclu.
Ce reste dit si la répartition de Σ entre les directions d'un module compte, au-delà de son échelle moyenne.

## Montage
Llama-3.2-1B, tâche de format, rang 2, sept modules, écrêtage 1,0, α = r, les deux bras à **5·10⁻³** (leur taux retenu), sur
**15 graines fraîches, 50 à 64**, jamais utilisées. Appariement par graine.

## Règles, fixées d'avance
On note d = `top_sigma` − `top_scalar`, en points, par graine (15 différences).
- **Établi** : d > 0 et t > **2,145** (bilatéral, 14 d.l.).
- **Équivalence** : l'intervalle de confiance à 90 % de d est entièrement dans **[−1 ; +1] point** (marge fixée ici, avant les runs).
- Sinon : **non tranché** ; on rapporte la moyenne, t et l'intervalle.

## Ce que chaque issue voudra dire
- établi : l'échelle de Σ agit comme un taux PAR DIRECTION, pas seulement par module ;
- équivalent : l'échelle moyenne par module suffit ; la répartition interne de Σ ne compte pas, à un point près ;
- non tranché : écrit tel quel.

## Engagements
Rapporté quelle que soit l'issue ; aucune graine ajoutée.
