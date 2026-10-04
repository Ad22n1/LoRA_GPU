# Additif à `PROTOCOL_long_training.md` — relance des runs de sélection tués

Écrit et committé AVANT la relance ; sa date est celle de son commit. Il ne change ni les bras, ni les taux, ni les graines, ni les tests.

À la fin de l'étape de sélection, 37 runs sur 40 étaient terminés ; trois (`top` à 10⁻² et 2·10⁻², graine 1 ; `top_unfrozen` à 2·10⁻³,
graine 0) étaient encore marqués « RUNNING » alors qu'aucun job ne restait dans la file : ils ont été tués sans écrire de statut final
(limite de temps ou perte de machine), ce qui est une panne, pas un résultat.

Règle, fixée avant la relance : ces runs sont marqués « FAILED » (le script refuse si un job reste dans la file), puis **relancés une seule
fois, à l'identique**, épinglés sur RTX 4000 Ada. Un run qui échoue encore laisse son bras **incomplet** ; il n'y a pas d'autre relance.
Le rapport (étape 2) ne part qu'après cette relance.
