# Rubik-Tensor : features à ajouter pour l'expérience « méta-tokens »

Principes directeurs, valables pour toutes les features :
- Rubik-Tensor reste une **bibliothèque d'environnement** : pas de code d'entraînement (torch.optim, boucles de training, checkpoints) dans le dépôt. Les objets « protocoles » sont des abstractions d'environnement et de données, consommées par le dépôt d'expériences.
- Toute API doit exister en version **batchée** dès la première implémentation : le raffinement a posteriori d'une classe mono-cube (comme `Cube` actuel) coûte plus cher qu'une généralisation immédiate (`B=1` n'est qu'un cas particulier).
- Toute fonctionnalité doit être **testée unitairement** (le repo a déjà `tests/` et CI : chaque feature s'accompagne de ses tests) et **benchmarkée** (`pytest-benchmark` ou script dédié, résultats archivés).

---

## Feature 1 : Environnement multicube (batches de cubes)

**Description.** Aujourd'hui, `Cube` porte un seul état `state` 1D de longueur 6·N², et `rotate_once` applique `torch.gather(self.state, 0, action)`. L'expérience exige d'appliquer simultanément des mouvements différents à des dizaines de milliers de cubes (génération de données, rollouts GRPO, beam search, DAgger). Il faut une classe `CubeBatch` (ou généralisation de `Cube`) dont l'état est un tenseur `(B, 6·N²)` et qui supporte les moves par-ligne.

**Definition of Done.**
- [ ] `CubeBatch(size, batch)` expose : état `(B, 6N²)` sur device/dtype paramétrables (int64 CPU, float32/bool GPU).
- [ ] `move(actions)` reçoit des indices de moves `(B,)` (ou `(B, T)` précalculés) et met à jour l'état batché en un seul `gather` sur `dim=1`, sans boucle Python sur le batch.
- [ ] `is_solved()` renvoie un masque booléen `(B,)` ; `solved_fraction()` un scalaire.
- [ ] Permutations : chaque move du batch est appliqué à sa ligne ; test exhaustif : pour B=1000 états et moves aléatoires, résultat identique à l'application séquentielle de `Cube` mono (`torch.equal`).
- [ ] Parité de performance : benchmark documenté du débit en moves/s (CPU et GPU) ; l'application de 10⁶ moves à un batch doit prendre < 1 s sur GPU.
- [ ] `Cube` mono reste utilisable sans breaking change (déléguer ou sous-classer ; sa doc précise qu'il fournit le sucre syntaxique sur `CubeBatch` avec B=1).
- [ ] Hors-sujet explicité : le modèle et la loss ne sont pas dans ce dépôt.

**Draft de résolution.** `state` devient `(B, L)` avec `L = 6·N²`. Le tenseur d'actions `actions[axis, slice, inverse]` (déjà précalculé, longueur `L`) est aplatï en une table `PERMS[n_moves, L]`. Un batch de moves d'indices `idx (B,)` donne `perm = PERMS[idx]` de forme `(B, L)`, et `new_state = torch.gather(state, 1, perm)`. Bien vérifier la sémantique du `gather` : `perm[dst]=src` signifie `new[dst]=old[perm[dst]]`, le code existant (`torch.gather(state, 0, action)`) estAlready correct pour le cas 1D, la transposition en `dim=1` est directe. Ajouter une mise en page optionnelle « flat index » : encoder l'état en `(B, L)` int64 mais garder une version `uint8` pour la mémoire sur très grands batchs. Le support float32 GPU existant (`Cube.to`) doit marcher sur le batch (le `gather` sur float est équivalent par permutation).

---

## Feature 2 : Compatibilité avec le vocabulaire standard (notations faces)

**Description.** Les moves actuels sont en notation géométrique `X1 Y2i...` (axe, tranche, inverse), parfaite pour la généralisation N×N×N mais étrangère à toute la littérature du domaine (DeepCubeA, EfficientCube, CayleyPy, datasets publics : notation faces U D L R F B, métrique HTM) et non canonique pour construire l'alphabet BPE de l'expérience. Il faut une couche de correspondance bidirectionnelle par taille de cube, y compris les variants avec méta-tokens.

**Definition of Done.**
- [ ] `standard_moves(size) -> list[str]` : notation standard par taille — 2×2 et couche externe du 3×3 : `{U, U', D, D', L, L', R, R', F, F', B, B'}` ; variant étendu `{..., U2, D2, ...}` en option.
- [ ] `to_internal(move_str) -> (axis, slice, inverse)` et `from_internal(triple) -> move_str`, exacts et réciproques (tests par aller-retour exhaustif sur toutes les tailles 2 à 5, ou du moins sur 2, 3, 4 pour chaque tranche externe).
- [ ] Gestion des conventions ambiguës : la face U au sens standard est-elle la tranche Y la plus haute en coordonnées internes, et le sens horaire vu de la face correspond-il à `inverse=0` ? Trancher par test physique : `U` appliqué au cube résolu doit déplacer la face avant vers la gauche (vérifié visuellement et par test unitaire sur des paires (sticker source, sticker destination) imposées).
- [ ] Parse de séquences : `parse_standard("R U R' U'")` → liste de moves internes ; erreurs explicites sur notation inconnue.
- [ ] Méta-tokens : un type `MoveVocabulary` encapsule (base moves standard + méta-tokens chargés) avec `encode(sequence) -> indices`, `decode(tokens) -> mouvement de base standard`, `compose(token)` → permutation du token via `compose_moves` (précalcul et cache, un méta-token = une permutation précomposée s'appliquant en un seul `gather`).
- [ ] Le test de non-régression clé : un corpus de 100 séquences standard aléatoires, encodées en interne puis reconverties, reconstitue exactement les mêmes états successifs (`torch.equal`).

**Draft de résolution.** La correspondance dépend de la taille : pour un cube N, `U = (axe Y, tranche size−1, sens vu du dessus)`, `D = (Y, 0, sens inverse)`, etc. La subtilité est le **sens horaire** : un quart de tour d'une face est défini vu depuis l'extérieur de la face, alors que `inverse` est homogène pour tout le cube (pour l'axe Y, U est le sens inverse de D). Ne pas deviner : écrire un test d'identité sur deux versions (une hand-checked au 3×3 par exemple une roadmap de Kociemba, l'autre dérivée de la convention de rotation POS_ROTATIONS du code actuel) et corriger la table jusqu'à identité. Fournir deux modules : `rubik/notation.py` (pur, sans torch) pour les correspondances de chaînes, et `rubik/vocabulary.py` pour l'encapsulation méta-tokens (fallback si un méta-token n'est pas présent lors du décodage, util `to_standard_sequence(tokens)`).

---

## Feature 3 : Solveurs, oracle de distance et générateurs de labels

**Description.** Toute l'expérience repose sur des états associés à des solutions optimales (ou quasi optimales) et des distances exactes : SFT (solutions optimales), DAgger (solution optimale depuis tout état intermédiaire), évaluation (taux d'optimalité), stratification des données par distance. Le repo doit fournir ces oracles, du moins pour 2×2, et encapsuler des solveurs externes pour les tailles supérieures.

**Definition of Done.**
- [ ] **BFS complet 2×2** : construction d'une table de distance exhaustive (par canevas : une face fixée en couleur, orientations du cube global éliminées par canonisation des couleurs), quelques millions d'états, soit en mémoire (≈ 50 Mo en uint8) soit sur disque avec mmap.
- [ ] `optimal_distance(state) -> uint8`, vectorisée : distances d'un batch `(B,)` en < 10 ms.
- [ ] `optimal_solution(state) -> str` : une solution optimale (par BFS inverse depuis l'état, avec règle canonique documentée pour casser les ex æquo : ordre lexicographique sur les symboles standard).
- [ ] **Wrapper solveur externe** (3×3+) : interface `Solver` avec deux implémentations, `Kociemba` (paquet `kociemba`, 3×3) et `dwalton` (NxN, périphérique), et `solve_batch` (multiprocessing, I/O isolé) — les API tierces ne doivent jamais être appelées dans la boucle GPU.
- [ ] Tests : distance du cube résolu = 0 ; distance d'un état après un move optimal = ±1 ; depuis 10⁴ scrambles générés, la longueur de la solution optimale correspond à la table de distance.
- [ ] Benchmarks et volumétrie documentés : temps de construction de la BFS 2×2, empreinte mémoire, débit de `solve_batch`.

**Draft de résolution.** Le BFS 2×2 se fait sur l'espace quotienté : canoniser l'état en fixant la couleur d'un coin (6 orientations du globe × 24 rotations du repère se réduisent à fixer deux stickers adjacents, cf. quotient par sims du groupe, environ 3,7 millions de classes × 2⁷ orientations = ~3,6M d'états réels / quotient à vérifier dans l'implémentation). Encodage compact : pour un cube 2×2 (8 cubies × 3 orientations), l'état entier tient en 8 × 3 bits = 24 bits, un entier par état : le BFS est un parcours sur les entiers via des tables de translation précalculées à partir du tenseur d'actions existant. Les distances sont stockées en uint8. La solution optimale se retrouve en marchant en descente de gradient strict (distance−1) depuis l'état jusqu'à 0, avec choix lexicographique canonique parmis les moves de descente : linéaire, pas de second BFS.

---

## Feature 4 : Générateurs de données avec stratification et anti-fuite

**Description.** Les protocoles SFT, DAgger et GRPO consomment chacun un type de données différent : couples (état, solution) ; couples (état intermédiaire, solution optimale) ; états de départ répartis par distance. Le générateur doit être streaming (le volume cible va jusqu'à 10⁷–10⁸ exemples), stratifié par distance à l'état résolu, et garantir l'absence de collision entre corpus.

**Definition of Done.**
- [ ] `ScrambleGenerator(size, distance_range, distance_sampler)` : produit des batches d'états avec la distance **exacte** connue (par rejection sampling via l'oracle de Feature 3 sur 2×2, par tranches de marche sur 3×3 avec labels approximatés explicitement comme tels).
- [ ] `DatasetRecipe` : type de sortie paramétrable — format SFT (tokens), format GRPO (état seul + distance + solution de référence éventuelle), format DAgger (couple état/solution). Interface `iter_batches(batch_size, device)` en streaming à la volée (pas de matériel disque obligatoire).
- [ ] Anti-fuite : un `KeySet` (fROOLE via hash bytes de l'état canonique) partagé entre corpora ; le générateur saute tout état d'un KeySet déclaré (test : 100 % de disjointness vérifié par intersection vide sur 10⁶ états).
- [ ] Statistiques d'échantillon exposées : histogramme des distances, taille effective, temps de génération par million.
- [ ] Les graines sont explicites et reproductibles (même graine → même corpus exact).

**Draft de résolution.** Tirage par distance exacte : sur 2×2, l'approche efficace est d'inverser le BFS — échantillonner un état à distance d en tirant d'abord d uniformément sur la plage cible, puis en partant d'un état de distance d (table indexée par distance), ce qui évite le rejection sampling coûteux. Génération par torch : précalculer la liste des états par distance au moment de la construction de l'oracle (Feature 3), puis échantillonner un état avec des indices aléatoires de torch (rapide, sur GPU si besoin). Sur 3×3, la stratification est approximative (marches de longueur d, en précisant que la distance réelle est ≤ d) : documenter le biais et l'exclure du calcul des distances effectives (l'oracle true est inaccessible en volume, seul le test de validation du résultat est exact).

---

## Feature 5 : Moteur de rollout (boucle d'interaction modèle ↔ environnement)

**Description.** L'évaluation, DAgger et GRPO ont tous besoin de la même brique : générer en batch des séquences de moves/méta-tokens depuis un modèle (au sens large : fonction de tirage de tokens), les appliquer aux cubes, détecter les résolutions, couper ou abandonner selon une borne. Le repo doit fournir ce « moteur de rollout » pur environnement (le modèle n'est qu'un callable fourni par l'appelant, ce qui maintient la frontière bibliothèque/protocole).

**Definition of Done.**
- [ ] `RolloutEngine(cube_batch, vocabulary, max_tokens)` avec l'API : `run(policy, n_steps) -> RolloutResult`.
- [ ] `policy(prev_tokens, states) -> (token_indices, log_probs ou None)` : duck typing, sans dépendance au module modèle ; un simple `argmax` des logits suffit à l'évaluation.
- [ ] Contraintes intégrées : masque de vocabulaire (couleurs/START exclus), rejet d'inverse du dernier token (paramétrable, ex. `no_inverse=True`), borne de tokens, arrêt anticipé sur DONE.
- [ ] Détection de résolution à chaque token émis, en batch (vectorisée via `is_solved` sur tout le batch chaque étape, ou via un comptage d'étapes en avance de phase).
- [ ] `RolloutResult` : tenseurs (tokens émis, masque de fin, cubes résolus, nb de tokens avant résolution, longueurs de solutions émises) + en-tête de sérialisation JSONL (`{state, tokens, moves, solved, length, walltime}` — le format du plan d'expérience, section 6.2).
- [ ] Tests : sur un policy qui suit un scramble inversé exact, le moteur détecte 100 % de résolutions au bon token ; la borne d'abandon fonctionne ; le rejet d'inverse ne casse pas les solutions qui en ont légitimement besoin (test avec solution contenant un inverse légitime... le rejet n'exclut que l'inverse immédiat).
- [ ] Débit : ≥ 10⁴ rollouts/s en batch de 10⁴ cubes et 20 tokens max sur GPU, le goulot devant être le modèle, pas l'environnement (benchmark à l'appui, policy = linear/no-op).

**Draft de résolution.** L'état du rollout est : `states (B, L)`, `last_token (B,)`, `active (B,)` (masque des cubes non résolus et sous la borne). À chaque étape : le moteur calcule le masque autorisé par vocabulaire (`vocab_mask` précalculé `(V,)`) et le masque d'inverse immédiat (`inv[last_token]`), appelle `policy`, applique les tokens via la table `compose` de la Feature 2 (un méta-token = une permutation précomposée, un seul gather par token). `is_solved` est appelé sur tout le batch à chaque étape (peu coûteux, un seul `torch.all` sur sous-tenseurs). Pour éviter d'échantillonner sur des cubes résolus,ophoniques : mettre `last_token` à DONE pour les cubes inactifs et masquer leur contribution. Le moteur ne gère ni reward ni logique de confiance : ces éléments sont la Feature 6.

---

## Feature 6 : Objets des protocoles d'entraînement (SFT, DAgger, GRPO)

**Description.** Le repo ne contient pas le training, mais fournit ses **objets d'environnement et de données**, sans importer torch.optim pour autant. Chaque protocole du plan d'expérience se réduit à un assemblage des Features 1–5, plus trois briques spécifiques décrites ici.

**Definition of Done.**

- [ ] **6a. SFT** : `SFTDataset` (via `DatasetRecipe` de la Feature 4) qui émet des tenseurs prêts à la consommation : `tokens (B, T)`, `labels (B, T)` avec −100 sur l'état (les couleurs ne comptent pas dans la loss), padding batché à la longueur max, et `collate` compatible avec le sampler de curriculum (pondérations par distance ajustables sans recharger le data). Le repos ne fait **pas** la loss.
- [ ] **6b. DAgger** : `IntermediateStateOracle` qui, depuis un rollout fourni par la Feature 5, extrait les états intermédiaires (au token près) et sollicite l'oracle de la Feature 3 pour les solutions optimales correspondantes (construction d'un dataset de correction par batch). Test de correction : sur un rollout intentionnellement détourné de l'optimal, l'oracle produit des solutions qui résolvent bien depuis chaque état intermédiaire (assertion `is_solved` après application).
- [ ] **6c. GRPO** : `GRPOEnv` avec `reset(states)` et `step(token_indices) -> (active, reward, info)`. Reward paramétrable via un objet `RewardFn` : `BinarySolveReward` (1 si résolu avant la borne), `LengthPenalizedReward` (1 − α·(longueur émise/longueur attendue), seulement si résolu, sinon 0), avec l'alpha et la longueur attendue (distance oracle) fournis à la construction. Le reward est calculé en batch (`(B, G,)`), aucune référence à un algorithme d'optimisation (advantage, KL) dans ce dépôt.
- [ ] **6d. Évaluation** : `benchmark(policy, test_states, vocabulary, mode)` retourne le tableau de métriques standard du plan (solve rate global et par bucket de distance, optimalité relative, tokens émis par solution, walltime), et la sérialisation JSONL du plan (section 6.2), prête à être archivée.
- [ ] Tests : reward binary correctness sur des rollouts connus (résolus/non résolus fabriqués), test de distance dans reward de longueur émis avec un alphaTrace covarié, aucun test de convergence (interdits par la frontière du scope).

**Draft de résolution.** 6a est quasi gratuit si la Feature 4 est bien conçue (le format SFT n'est qu'une recette + un collate). 6b est le plus subtil : les états intermédiaires existent déjà dans `RolloutResult.states` si le moteur de la Feature 5 archive la série des états (option `capture_states=True`), sinon il faut les reconstituer en rejouant la solution (rejeu = gather batché, pas coûteux). 6c est une surcouche fine du RolloutEngine dont la seule logique additionnelle est le calcul du reward à la fin : tout peut se réduire à une fonction pure `(emis, résolus, distances) -> reward`. 6d n'est qu'une agrégation de `RolloutResult` avec jointure sur les distances de l'oracle.

---

## Feature 7 : Canonisation par symétries (accélérateur transverse)

**Description.** Le groupe du cube a 48 symétries (rotations + réflexions du repère, × symétries des couleurs une fois une face de référence fixée du 2×2 ou du 3×3). CayleyPy et CubeGPT-3 exploitent ces symétries pour réduire l'espace d'états d'un facteur ~48, diminuant le volume de données nécessaires et stabilisant les invariances apprises. Maya est une optimisation importante pour le modèle, mais c'est aussi un incrément pour l'oracle (Feature 3) et le générateur (Feature 4).

**Definition of Done.**
- [ ] `canonize(states) -> (canon_states, symmetry_ids)` : représentation canonique vectorisée, plus l'identifiant de la symétrie appliquée (pour pouvoir décanner).
- [ ] `decanonize(states, symmetry_ids)` inverse exact (test d'aller-retour exhaustif sur B=10⁵ états aléatoires).
- [ ] Concaténation avec l'oracle : la table de distance BFS est construite sur les états canoniques (réduction ~48× du temps de construction et de la mémoire) et les distances publiques restent exactes.
- [ ] Preuve par test : deux états liés par une symétrie quelconque donnent le même état canonique ; les distances sont identiques ; la solution canonique décannonisée résout le cube d'origine.

**Draft de résolution.** Les 48 symétries s'appliquent comme des transformations affines (rotation du repère + permutation des faces) sur les coordonnées du tenseur sparse existant : elles se précalculent en 48 tables de permutation d'indices stickers (à partir des mêmes primitives `POS_ROTATIONS` / `FACE_ROTATIONS` du module d'actions actuel, ce qui limite le code nouveau). La canonisation est le min lexicographique des 48 états permutés — 48 gathers batchés puis un `min` sur la première dimension, environ 50× le coût d'un move mais réalisé une fois par état, pas par move. Pour 2×2 la réduction est d'environ 24× en comptant seulement les rotations du repère, 48× avec les réflexions (à documenter : les réflexions ne sont pas des isométries du sens des moves, elles inversent les moves primés, ce qui est pertinent seulement si les solutions sont canonisées aussi ; sinon limiter aux 24 rotations).

---

## Synthèse et ordre d'implémentation

| # | Feature | Dépend de | Priorité |
|---|---|---|---|
| 1 | CubeBatch (multicubes) | — | **P0**, tout en dépend |
| 2 | Notation standard + `MoveVocabulary` | 1 | **P0**, l'alphabet BPE en dépend |
| 3 | Oracle BFS (2×2) + solveurs externes | 1 | **P0**, features 4, 6 en dépendent |
| 4 | Générateurs stratifiés + anti-fuite | 1, 3 | P1 |
| 5 | Rollout engine | 1, 2 | P1 |
| 6 | Objets protocoles (SFT/DAgger/GRPO/eval) | 1–5 | P1–P2 |
| 7 | Canonisation par symétries | 1 | P2 (optimisation, précieuse mais différer si besoin) |

Ordre de attack recommandé : **1 → 2 → 3** constituent le socle minimal pour commencer les expériences E1 du plan (SFT sur solutions optimales) ; la feature 4 et 5 suivent pour l'évaluation complète ; la 6 pour GRPO ; la 7 en optimisation transverse quand un premier passage de l'expérience est joué.

**Frontière de responsabilité à écrire dans le README** : Rubik-Tensor fournit environnement, notation, oracle, données et moteur de rollout. Le dépôt d'expériences (à créer) fournit modèle, loss, optimizer, GRPO algorithmique et bookkeeping. Le contrat entre les deux est la duck typing `policy` (Feature 5) et les tenseurs de données (Feature 6).
