# Plan d'expérience : méta-tokens et résolution du Rubik's Cube

## 1. Objectif

**Question centrale** : l'introduction de méta-tokens (tokenization BPE des séquences de mouvements) améliore-t-elle les performances d'un modèle autoregressif entraîné à résoudre un Rubik's Cube ?

**Hypothèse principale (H1)** : à budget d'entraînement égal (mêmes exemples, même budget de calcul, même nombre de paramètres), un modèle entraîné sur des solutions compressées par méta-tokens atteint un taux de résolution et une optimalité supérieurs à un modèle entraîné sur les mouvements de base.

**Mécanisme supposé** : la compression raccourcit l'horizon de prédiction (ex. 7,6 → 3,6 symboles par solution à 800 merges sur le 2×2), ce qui (a) réduit les erreurs composées de la génération autoregressive, (b) densifie le crédit d'apprentissage par solution, (c) réduit la longueur des rollouts en RL.

**Hypothèses secondaires** :
- H2 : le gain est plus marqué sur les scrambles longs (distance élevée) que courts.
- H3 : le BPE redécouvre des motifs structurellement connus (commutateurs, séquences F2L-like), et ces tokens sont associés à un gain mesurable versus des tokens aléatoires de même longueur.
- H4 : le gain persiste (ou croît) en post-training RL (GRPO), via des rollouts plus courts et un reward atteint plus tôt.
- H5 (exploration) : le pré-entraînement du modèle sur les séquences de méta-tokens « hors contexte » (modèle de langage des solutions seules) accélère la convergence.

**Résultat attendu** : documenta de référence publiable si H1 confirmée : à notre connaissance, l'action chunking par BPE n'a jamais été appliqué au Rubik's Cube.

## 2. Données

### 2.1 Protocole de création

Le cube offre un générateur parfait : toute marche aléatoire depuis l'état résolu produit un couple (état, solution) gratuitement — la solution est l'inverse de la marche.

**Constitution du corpus** :
1. Générer des marches aléatoires depuis l'état résolu, en interdisant l'inverse du mouvement précédent et la troisième répétition (X X = X'i est factorisé).
2. La solution = séquence inversée de la marche.
3. **Optimalisation** : pour le 2×2, calculer la distance optimale de chaque état par BFS complet de l'espace d'états (3,7 millions d'états à couleurs fixes ; ~10⁷ avec orientations du cube entier si non canonisées). Pour le 3×3, utiliser un solveur (Kociemba via `kociemba` ou `dwalton76/rubiks-cube-NxNxN-solver`) : les solutions ne sont pas optimales mais courtes et cohérentes ; l'optimalité stricte est vérifiable via dataset DeepCubeA seulement.
4. Variante « sous-résolutions » : inclure des paires (état source → état cible) aléatoires, Cible ≠ état résolu (comme dans « Solving Rubik's Cube Without Tricky Sampling ») pour un signal plus dense et une entrée au GRPO progressif.

### 2.2 Volume

| Cube | Corpus BPE | Train | Val | Test |
|---|---|---|---|---|
| 2×2 | 200 000 solutions | 5–20 M exemples (générés à la volée) | 50 000 | 114 606 états (test standardisé du domaine cf. arXiv 2512.03400) |
| 3×3 | 500 000 solutions | 20–100 M exemples | 100 000 | 100 000 états + dataset de test DeepCubeA |

Le BPE se fit une fois sur le corpus, avant le training ; ne jamais recycler les solutions de test dans le corpus BPE.

### 2.3 Diversité de complexité

- Échantillonner la longueur de scramble uniformément sur [1, d_max] (d_max = 14 pour le 2×2 dont le diamètre est 11 HTM en quart-de-tour ; ~20 HTM pour le 3×3).
- Stratification par distance à l'état résolu (buckets de distance 1–2, 3–5, 6–8, ...) avec rapport contrôlé, pour que les états faciles ne dominent pas le corpus (distribution uniforme sur les distances ≈ la marche aléatoire surpondère les distances moyennes).
- Courriculum optionnel : commencer par d ≤ 6, étendre progressivement vers d_max.

### 2.4 Exhaustivité et contrôle de fuite

- 2×2 : vérifier la couverture du corpus (toutes les distances représentées ; les états à distance k arrivent par plusieurs marches différentes).
- Les états de test sont générés avec une graine disjointe.
- Contrôle anti-triche : un état de test ne doit pas apparaître dans le train (lookup par hash d'état canonique ; sur 2×2 c'est vérifiable exhaustivement).

### 2.5 Mécanisme de factorisation en séquences optimales

- **Factorisation off-line** : aplatir les répétitions immédiates (X X X → X'i, X X → X2 si alphabet étendu), puis réduire par annulation (X X'i → ε).
- **Optimalité garantie (2×2)** : BFS bidirectionnel donnant LA plus courte séquence ; pour casser les ex æquo, choisir canoniquement (ordre lexicographique sur les symboles, ou plus grand tokens BPE préférentiellement, testé en ablation H3-bis).
- Attention : le choix de la solution parmi les solutions optimales équivalentes est un biais de corpus ; documenter la règle et tester sa sensibilité (ablation : solution canonique vs solution aléatoire parmi les optimales).

## 3. Action chunking (méta-tokens)

### 3.1 Alphabet de base des mouvements

- **Cellule de référence (2×2)** : 12 mouvements de base U U'i D D'i L L'i R R'i F F'i B B'i (quarter-turn metric, notation faces standard, nécessaire pour comparer à la littérature).
- Variantes en ablation : 18 mouvements (incluant demi-tours X2) ; notation axe/tranche (X1, Y2i...) native de Rubik-Tensor avec table de correspondance.
- Symbole final de fin de solution (DONE/STOP) explicitement dans l'alphabet pour permettre au modèle de s'arrêter.

### 3.2 Tokenizer de base

- **BPE** standard (fusion de la paire la plus fréquente, ordre d'apprentissage conservé pour l'encodage) : implémenté et validé, compression mesurée 1,07× (10 merges) à 2,13× (800 merges) sur corpus 2×2.
- **Ablation** : tokenisation unigram (tête de SentencePiece) ; avantage théorique de fournir le meilleur voisin en cas de couverture imparfaite.
- L'encodage doit être déterministe, décodable en mouvements de base sans perte (roundtrip vérifié), et fermé sous composition.

### 3.3 Tailles de vocabulaire cible

Courbe de compression : 50, 200, 800, 2 000 merges (mesurée sur le 2×2 : 1,41×, 1,76×, 2,13×, ~2,5×). Chaque taille = une branche de l'expérience + baseline sans BPE. La taille optimale attendue résulte d'un compromis : vocabulaire plus gros = séquences plus courtes mais tokens plus rares (moins de contexte d'apprentissage par token, cf. scaling laws d'EfficientCube).

### 3.4 Identification des méta-tokens qui sont des mouvements connus

Protocole d'analyse :
1. Classer tous les méta-tokens par fréquence × réduction de longueur.
2. Décoder chacun en moves de base et calculer ses propriétés de permutation : points fixes, effet sur les couches, forme de cycle.
3. Matcher contre une base de référence de patterns connus : commutateurs simples [A,B] = A B A'i B'i, conjugaisons A B A'i, séquences d'insertion F2L-like de la méthode CFOP, et leurs équivalents symétriques.
4. Métrique : fraction du vocabulaire identifiée comme patterns structurels, et part de probabilité de ces tokens dans le modèle entraîné.
5. Test de vérification H3 : remplacer les méta-tokens « structurels » par des tokens de longueur équivalente assemblés aléatoirement (même compression, aucun sens) et observer la différence de performance.

## 4. Modeling

### 4.1 Architecture

- **Modèle** : GPT decoder-only (transformer autoregressif causal) sur Rubik-Tensor (environnement batché sur GPU).
- **Entrée** : séquence de tokens = [état du cube] + [START] + [tokens de solution]. Trois encodages d'état en ablation :
  - (a) 54 stickers (3×3) ou 24 (2×2) en tokens de couleur (baseline retenue à ce jour) ;
  - (b) stickers + quelques derniers moves (contexte d'action, cf. HF 2×2 solver) ;
  - (c) état canonisé par symétries du cube (réduit la taille effective de l'espace d'états de ~48×, cf. CayleyPy / CubeGPT).
- **Sortie** : distribution sur le vocabulaire (couleurs + moves/méta-tokens + DONE).
- **Têtes auxiliaires** (ablation) : tête de valeur régressant la distance à l'état résolu (CayleyPy-style diffusion distance) en plus de la distribution de mouvements ; gain ×3,5 rapporté dans le modèle HF 2×2.
- **Tailles** : petit (25M, 8 couches, d=512) et moyen (~100M) pour tester l'interaction compression × échelle.
- **Initialisation des embeddings de méta-tokens** : composition des embeddings des moves constituants (moyenne ou somme), ablation contre initialisation aléatoire — ce qui teste la partie « prétrain de l'embedding » de votre idée initiale.

### 4.2 Mécanisme d'inférence

- **Génération contrainte** : seuls les tokens de mouvement (base ou méta) et DONE sont échantillonnables ; chaque token émis est décodé en moves de base, appliqué à l'état courant, et l'état résolu est testé après chaque token (vérification d'état gratuite).
- **Modes** : greedy (baseline), beam search (largeur 3–16, guidé par la tête de valeur le cas échéant), échantillonnage avec température.
- **Rejet d'inversion** : interdire le move/méta-token exactement inverse du dernier émis (règle déjà bénéfique à 1,6 % d'entraînement dans le HF 2×2 solver).
- **Borne de sécurité** : nombre maximal de tokens émis avant abandon (ex. 3× la longueur attendue).

## 5. Training

### 5.1 Étape 1 : SFT (Supervised Fine-Tuning) sur solutions optimales

- Objectif : cross-entropy de next-token prediction sur la séquence de solution (tokens couleur ignorés dans la loss).
- C'est le moteur principal : le modèle imite le corpus de couples (état, solution optimal/cohérent).
- Protocole : 3 000–10 000 steps, batch 256, AdamW 3e-4, warmup + cosine decay, early stopping sur la loss de validation. Mesurer la courbe loss vs. taux de résolution (le CE reste un proxy : la résolution greedy est la vraie métrique).
- Curriculum optionnel : ordonnancer les distances de scramble croissantes plutôt que sampling uniforme.

### 5.2 Étape 2 : DAgger (correction d'erreurs composées)

- Problème visé : en SFT le modèle n'apprend que sur des préfixes issus de solutions optimales ; à l'inférence il se retrouve dans des états « hors distribution » créés par ses propres erreurs.
- Protocole : rollouts du modèle courant sur des scrambles d'entraînement ; pour chaque état intermédiaire rencontré, solliciter un oracle (solveur / BFS) pour la solution optimale à partir de cet état, ajouter ce couple au dataset, réentraîner. 2–4 itérations DAgger.

### 5.3 Étape 3 : GRPO (post-training RL)

- Objectif : maximiser le taux de résolution au-delà de ce que l'imitation apprend ; c'est là que H4 (effet des méta-tokens sur le RL) se joue.
- Protocole : pour chaque état de training, échantillonner G rollouts (ex. G=8) depuis le modèle courant ; reward = 1 si le cube est résolu dans la borne de tokens, 0 sinon (reward vérifiable) ; avantage = reward centré-normalisé par groupe ; update PPO-clip sans critique, avec KL-pénalité vers le SFT.
- Variante de reward (ablation) : 1 − α·(longueur de solution émise / longueur attendue), pour pénaliser les solutions trop longues.
- Curseur spécifique à H4 : le GRPO démarre avec le modèle déjà résolvant en partie les états courts (sinon reward nul constant et échec du GRPO) ; les méta-tokens doivent faire apparaître des reward non-nuls plus tôt — c'est le levier testé.
- Alternatives si le GRPO bloque : label smoothing sur les étapes « vers la solution » pour donner un reward partiel (nombre de tokens BPE correctement prédits avant la première erreur de préfixe — c'est une mesure de plus proche préfixe, agit comme reward dense).

### 5.4 Étape 4 (option) : SFT inverse et world-model

- Prédire l'état bis (l'état après application du move) depuis (état, move), cf. arXiv 2512.03400 : le world model appris en prétraining prédit la qualité du GRPO. Loss jointe classification + next-token. C'est un tests secondaire de H5.

## 6. Évaluation

### 6.1 Benchmarks de test

- **2×2** : 114 606 états de test générés par scramble standard (graine retenue pour toute l'expérience), scrambles de distance uniforme sur [1, 14]. Ground truth optimal disponible via BFS complet.
- **3×3** : dataset de test DeepCubeA (scrambles classiques) pour comparabilité à la littérature (DeepCubeA 60,3 % d'optimalité, EfficientCube 69,6 %, CayleyPy 98,4 %) + 100 000 états aléatoires.
- Le test est figé avant tout entraînement et identique pour toutes les variantes (avec et sans BPE).

### 6.2 Format

Chaque cas de test = état de cube encodé (24 facelets 2×2 / 54 facelets 3×3) ; le modèle émet un flux de tokens décodé en moves de base ; le flux est évalué contre le vérificateur d'état Rubik-Tensor. Format de sortie archivé : JSON Lignes {state, tokens, moves, solved, length, optimal_length, walltime}.

### 6.3 Métriques

Principales :
1. **Solve rate** (% d'états résolus), par bucket de distance et global.
2. **Optimalité** (% de solutions de longueur optimale) et longueur moyenne excédentaire par rapport à l'optimal.
3. **Tokens émis par solution** (mesure l'effet compression à qualité égale).

Secondaires :
4. Efficacité d'échantillon (solve rate vs. nombre d'exemples vus) — pour tester H1 à budget égal, la comparaison se fait à budget de calcul fixé (steps × batch × longueur de séquence par exemple : les méta-tokens raccourcissent les séquences, donc le contrôle doit être en tokens traités, pas en steps).
5. Walltime d'inférence par résolution ; couverture du test pour la variante stoppée à budget fixe.

### 6.4 Plan d'ablation et significativité

| Expérience | Variable | Fixé |
|---|---|---|
| E1 | Vocabulaire : sans BPE / 50 / 200 / 800 / 2 000 merges | données, budget tokens, architecture |
| E2 | DAgger on/off, GRPO on/off | SFT de départ |
| E3 | Méta-tokens structurels vs. aléatoires équivalents (H3) | compression identique |
| E4 | Tête de valeur, encoding d'état (a/b/c), init. d'embeddings composée vs. aléatoire | vocabulaire 200 |
| E5 | GRPO avec vs. sans méta-tokens (H4) : solve rate à budget RL identique | SFT respectif |

- Trois graines minimum par cellule ; test de significativité : test exact/binomial sur les solve rates (n grand), intervalle de confiance rapporté à 95 %.
- Critère de succès global : E1 montre un solve rate supérieur significatif pour au moins une taille de vocabulaire, à budget de tokens traités égal, et le gain se transmet en E5 (GRPO).

### 6.5 Calendrier indicatif

1. Semaine 1 : environnement batché sur Rubik-Tensor + génération 2×2 + BFS optimal + corpus BPE.
2. Semaine 2 : E1 sur 2×2 (petit modèle), E4.
3. Semaine 3 : DAgger + GRPO (E2, E5), analyse de tokens (E3).
4. Semaine 4+ : réplication 3×3 sur solveur non optimal, comparaison DeepCubeA.
