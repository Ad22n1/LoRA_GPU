# Protocole pré-enregistré — la tendance selon le rang : l'isolation aux rangs 4 et 8 (OpenBookQA, Qwen2.5-1.5B)

Écrit et committé AVANT tout run ; sa date est celle de son commit.
**Arrêt des lancements : mardi 29/09 à 20 h.** Ce qui n'est pas parti à cette heure n'entre pas dans le rapport.

## Pourquoi
Sur OpenBookQA avec Qwen2.5-1.5B, l'isolation existe au rang 2 et au rang 16. Les rangs 4 et 8 donnent la tendance, et le rang où l'avantage de
geler A s'arrête sur la bande dominante.

## Budgets exacts (par couche) et LoRA libre de budget égal
| rang r | B gelé | LoRA libre de budget égal | écart | A gelé au rang r |
|---|---|---|---|---|
| 4 | 72 704 | **r = 2** : 82 432 | +13,4 % (r = 1 : −43,3 %) | +26,8 % |
| 8 | 145 408 | **r = 4** : 164 864 | +13,4 % (r = 3 : −15,0 %) | +26,8 % |
Le rang du libre est celui dont le budget est le plus proche en valeur absolue ; au rang 2, c'était r = 1 (+13,4 %) : la relation est la même.

## Montage (à chaque rang r ∈ {4, 8})
- **Bras** : `top`, `bottom` (B gelé, rang r) ; `top_unfrozen`, `bottom_unfrozen` au rang du LoRA libre **et** au rang r ; `dual_top`,
  `dual_bottom` (A gelé, rang r) ; `free` au rang de budget égal. **9 bras par rang, 16 en tout** : B entraîné au rang 4 sert à la fois de
  « rang r » au rang 4 et de « rang du libre » au rang 8, et n'est entraîné qu'une fois.
- Configuration reprise d'un run de référence d'OpenBookQA ; seuls changent le mode, le rang (α = r), le taux et la graine.
- **Taux** : cinq par bras (5·10⁻⁴ à 10⁻²), sélection sur les graines 0 et 1, rapport sur les graines 2 à 6 au taux retenu, un cran de prolongement.
- **Deux vagues de rapport** : d'abord les bras gelés et le LoRA libre ; **G = libre − gelé est alors calculé et committé** (`refs_rank48.json`),
  puis seulement les bras dégelés et à A gelé sont rapportés. G est ainsi fixé avant que les bras qu'il juge ne soient mesurés.
- Une seule carte, RTX 4000 Ada. **Environ 240 runs, ~22 PARTITION-heures.**
- **Facteurs non sauvegardés** : le quota personnel (30 Go, dont 28 déjà utilisés) ne le permet pas. Le lanceur vérifie ce quota avant chaque étape.
- **Hors test** : le LoRA libre au rang 1 à 2·10⁻³, graines 2 à 6, facteurs sauvegardés, pour l'analyse « où écrit le LoRA libre » sur OpenBookQA.
  Ces cinq runs sont ceux du réentraînement déclaré de `PROTOCOL_free_writes.md` (`launch_fw.sh`) : ils ne tournent qu'une fois, et ce sont
  les seuls à sauvegarder leurs facteurs (environ 5 Mo chacun).

## Tests, écrits d'avance (pour chaque rang et chaque bande)
1. **U** = dégelé (rang du libre) − gelé (rang r), contre 0,7·G : au-dessus → le gel explique l'essentiel ; au-dessous de 0,3·G → l'essentiel vient
   d'ailleurs ; sinon partagé. **Plancher : G ≥ 2 points**, sinon non applicable.
2. **Le gel seul** : dégelé (rang r) − gelé (rang r), apparié, **t > 3,495**.
3. **A gelé moins B gelé** (rang r), apparié, **t > 3,495**.
4. **L'écart entre bandes, orienté** : g_gelé = `bottom` − `top` (B gelé) ; g_autre = même écart avec B entraîné (rang r). Statistique
   d = signe(g_gelé) · (g_gelé − g_autre), **t > 2,776**, lu comme un rétrécissement **seulement** si |g_gelé| ≥ 2 points et |g_autre| < |g_gelé| ;
   si g_autre change de signe avec |g_autre| ≥ 2 points, c'est un **renversement**, rapporté comme tel. Même test avec A gelé à la place.
Rapportés : **U/G** et **I = G − U**, avec leurs intervalles (bootstrap apparié sur les graines, 10 000 tirages) ; taux retenus ; cartes.

## Prédiction, fixée d'avance et lue de façon descriptive
Avec les rangs 2 et 16 déjà mesurés sur cette paire : **G** et **A gelé moins B gelé** diminuent de façon monotone du rang 2 au rang 16, pour
chaque bande. Rapportée comme tenue, ou non, rang par rang.

## Règles communes
Une relance pour échec machine ; rapporté quelle que soit l'issue ; aucune graine ni aucun taux ajoutés hors le cran de prolongement.
