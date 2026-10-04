# Note au protocole « OpenBookQA au rang 2 » — écrite le 26/09

Le protocole (`PROTOCOL_obqa_rank2.md`) déclare une différence avec la campagne du rang 16 : son `random_ortho` n'aurait pas été
salé. C'est faux. Vérifié sur les runs le 26/09 : les 101 runs `random_ortho` au rang 16 portent tous `subspace_salted = True`,
la valeur par défaut de la configuration, que leurs grilles ne précisaient pas. Il n'y a donc aucune différence de sel entre
les campagnes. Le protocole committé n'est pas réécrit ; cette note le corrige.
