# Protocole pré-enregistré — C2 : réplication de C sur 40 graines fraîches

Écrit et committé le 25/09, **après** avoir lu C et **avant** tout run de C2.

## Pourquoi, et ce qui NE se fait PAS
C (`PROTOCOL_sigma_residual.md`, graines 50-64) a donné d = `top_sigma` − `top_scalar` = −0,27 point (t = −0,54), IC90
[−1,16 ; +0,61] : ni établi, ni équivalent à ±1 point — « non tranché ». **C est rapporté tel quel.** Son protocole interdisait
d'ajouter des graines : C2 n'est donc PAS une extension de C. C'est une étude nouvelle, sur des graines fraîches, **analysée
seule** ; ses graines ne sont jamais mises en commun avec celles de C pour un test.

## Taille, fixée par un calcul de puissance fait avant les runs
Avec l'écart type de d observé dans C (≈ 1,94 point), la chance d'établir l'équivalence à ±1 point si les deux bras sont
identiques serait de 19 % avec 15 graines, 74 % avec 30, **89 % avec 40**. D'où **40 graines, 65 à 104**, jamais utilisées.

## Montage
Identique à C : Llama-3.2-1B, format, rang 2, sept modules, écrêtage 1,0, α = r, les deux bras à 5·10⁻³, appariés par graine.

## Règles, fixées d'avance (39 d.l.)
- **Établi** : d > 0 et t > **2,023** (bilatéral).
- **Équivalence** : l'IC90 de d entièrement dans **[−1 ; +1] point**.
- Sinon : **non tranché**.

## Engagements
Rapporté quelle que soit l'issue, à côté de C. Une mise en commun de C et C2 n'est jamais un test : au plus une description.
