# Additif à `PROTOCOL_freezeA_rank16.md` — budget, lecture fixée d'avance, comparaison au LoRA libre

Écrit et committé avant tout résultat de cette campagne (le message du commit dit s'il précède aussi tout run). Il ne change ni les bras,
ni les taux, ni les graines, ni les tests : il fixe d'avance leur lecture.

1. **Budget.** Au rang 16 sur Qwen2.5-1.5B, les bras qui gèlent A (r × d_out par module) entraînent **26,8 % de paramètres de plus** que les
   bras qui gèlent B (r × d_in) : 10 321 920 contre 8 142 848 sur les 28 couches. L'écart vient des modules du MLP : `gate` et `up` ont un
   côté sortie près de six fois plus large (+483 %), tandis que `k`, `v` et `down` jouent en sens inverse (−83 %). Déclaré ici, rapporté avec
   les résultats.

2. **Lecture fixée d'avance.** Si le test (1) — A gelé moins B gelé — est établi pour les deux bandes sur une tâche, l'asymétrie tient au rang
   16 sur cette tâche. S'il ne l'est pas, elle est rapportée comme propre aux rangs bas, et le texte sur PiCa (qui gèle A) est limité au rang 2
   pour cette tâche.

3. **Rapporté sans test.** `dual_X` − `free` (le LoRA libre au rang 7, référence existante), pour chaque bande et chaque tâche. Cette
   comparaison n'est pas à budget égal : les bras qui gèlent A entraînent 27,8 % de paramètres de plus que le LoRA libre au rang 7.
