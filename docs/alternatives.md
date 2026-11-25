# Rubik-Tensor face aux alternatives Python

## Question

Rubik-Tensor, augmenté des features du cahier des charges (batching multicube, notation standard, oracle de distances, générateurs de données, moteur de rollout, objets de protocoles, canonisation par symétries), peut-il être considéré comme un framework state of the art pour la modélisation ML du Rubik's Cube ? Quelles sont les alternatives en Python, et que valent-elles face à lui ?

## Méthodologie

Revue des bibliothèques et frameworks Python référencés dans la littérature du domaine (DeepCubeA, EfficientCube, CayleyPy, etc.) et des projets communautaires actifs. Critères d'évaluation alignés sur les besoins de l'expérience « méta-tokens » : capacité de batch GPU, généralité en taille de cube, notation standard, disponibilité d'un oracle de distance/solution, infrastructure de données et de protocoles, maturité logicielle.

## Taxonomie : trois familles d'alternatives

1. **Bibliothèques de simulation et manipulation** — représentation du cube et moves, sans ML (rubik-tensor actuel est ici, magiccube, pycuber, pglass/cube).
2. **Solveurs classiques** — recherche exacte ou heuristique, non-ML (kociemba, dwalton76, Santa 2023).
3. **Frameworks ML expérimentaux** — code de recherche entraînant des réseaux à résoudre le cube (DeepCubeA, EfficientCube, CayleyPy, CubeGPT-3).

La thèse de ce document : aucun projet existant ne couvre les trois familles à la fois. Le futur Rubik-Tensor est le seul à viser l'union.

## Fiches par alternative

### 1. [DeepCubeA](https://github.com/forestagostinelli/DeepCubeA) (Agostinelli et al., 2019)

- **Ce que c'est** : le framework ML historique. Approximate value iteration sur des états générés en marches inverses, réseau cost-to-go (PyTorch), recherche A* pondérée batch en C++.
- **Pros** : référence bibliométrique absolue (code utilisé par toute la littérature ultérieure) ; datasets 3×3 publiques (devenus le benchmark standard) ; méthode validée sur plusieurs puzzles.
- **Cons** : code de recherche non maintenu (Python 2 → portage partiel, style TF1 initial) ; batchs de recherche en C++ custom, opposé à un tween PyTorch ; aucune séparation environnement/protocole ; pas de batching GPU natif de l'environnement ; 3×3 et puzzles symétriques seulement, pas de N×N générique.
- **Vs future Rubik-Tensor** : Rubik-Tensor remplace la brique environnement par du PyTorch natif GPU-batché ; le benchmark de comparaison reste le dataset DeepCubeA, qui doit être consommé comme entrée de test (dans les features du cahier des charges, la grille d'absorption de ce dataset est un bon critère de validation d'interopérabilité).

### 2. [EfficientCube](https://github.com/kyo-takano/EfficientCube) (Takano, TMLR 2023)

- **Ce que c'est** : auto-supervision sur mélanges aléatoires, régression de la valeur, 69,6 % d'optimalité 3×3 avec beaucoup moins de données et de calcul que DeepCubeA.
- **Pros** : code net, notebooks reproductibles, datasets DeepCubeA embarqués en copies accessibles ; étude de scaling law très pédagogique ; architecture simple à réimplémenter.
- **Cons** : pas un framework — un dossier d'expérience one-shot ; environnement non batché (numpy pur, cpu-only) ; pas de notation standard à bord ; pas de couche RL/GRPO ; 3×3 figé.
- **Vs future Rubik-Tensor** : EfficientCube reste l'expérience de référence à répliquer pour valider le futur Rubik-Tensor (répliquer ses chiffres avec les mêmes données serait une validation forte à inclure dans le plan), mais son environnement ne passe pas à l'échelle du GRPO batch.

### 3. [CayleyPy](https://github.com/cayleypy/cayleypy) + [cayleypy-cube](https://github.com/khoruzhii/cayleypy-cube) (Chervov et al., NeurIPS 2025 spotlight)

- **Ce que c'est** : le SOTA actuel. Bibliothèque générale de graphes de Cayley (cayleypy) + application Rubik's Cube entraînant ResMLP de diffusion-distance + beam search GPU-accéléré. 98,4 % d'optimalité 3×3, solveurs ML 4×4 et 5×5 premiers du genre, dominance des résultats Kaggle Santa 2023.
- **Pros** : le plus proche d'un « framework de recherche » réel — générateurs, datasets, modèles, entraînement, benchmarks intégrés ; multi-puzzles par graphes de Cayley ; code actif et性能 oriented.
- **Cons** : optimisé pour un paradigme précis (heursée apprise + beam search), pas pour la génération autoregressive / les LLM — pas de tokenisation, pas de rollout modèle, pas d'interface policy ; abstraction « graphe de Cayley » moins immédiate pour les manipulations de cube (pas de facelets, pas d'interface visuelle) ; pas de couche RL.
- **Vs future Rubik-Tensor** : c'est le concurrent le plus sérieux sur le terrain « framework ML pour le cube ». Le futur Rubik-Tensor s'en distingue par trois angles : (a) environnement orienté modèles séquentiels (vocabulaire, méta-tokens, rollout autoregressif), terrain que CayleyPy ne couvre pas ; (b) généralité des tailles avec interface tensorisée simple ; (c) séparation stricte bibliothèque/protocole, quand CayleyPy mélange puzzle et algo. Mentionner honnêtement que CayleyPy reste devant sur la recherche heuristique (beam search) : le futur Rubik-Tensor ne cherche pas à rivaliser sur ce point, mais à ouvrir le champ action-sequencing que CayleyPy n'adresse pas.

### 4. [CubeGPT / CubeGPT-3](https://github.com/stanleytheli/cubegpt-3) (communautaire)

- **Ce que c'est** : le projet communautaire le plus proche thématiquement de Rubik-Tensor futur — transformers pour résoudre le 3×3, canonisation par symétries, recherche hybride, scaling to 200M échantillons.
- **Pros** : démonstration de faisabilité complète de l'approche transformer sur cube ; techniques transposables canonisation, précalcul sur GPU du pipeline de données.
- **Cons** : projet solo non packagé (pas de lib, pas d'API, `main.py` monolithique) ; pas d'environnement batché ; pas de RL ; choix de design documentés hors repo ; 3×3 only.
- **Vs future Rubik-Tensor** : CubeGPT valide la demande ; Rubik-Tensor futur书目 le formalise en framework. C'est l'exemple type du « code d'expérience » qui justifie l'existence d'une couche bibliothèque propre.

### 5. [magiccube](https://github.com/trincaog/magiccube) (trincaog)

- **Ce que c'est** : cube N×N×N pur Python avec recherche de solution incluse (méthodes Kociemba-like), inverses en fonction de l'historique, interface simple.
- **Pros** : le plus propre des outils de simulation/manipulation en Python ; N×N×N natif ; tests.
- **Cons** : CPU only, aucune considération ML (pas de tenseurs, pas de batch) ; perfs limitées ; notation propre.
- **Vs future Rubik-Tensor** : sur le terrain simulation, Rubik-Tensor actuel le surpasse déjà en vitesse (moves précalculés en permutations + gather vs re-calcul de permutations à chaque opération). magiccube reste utile comme oracle de référence (comparaison d'équivalence d'états après moves standard pour tester la Feature 2).

### 6. [pycuber](https://github.com/adrianliaw/PyCuber) et [pglass/cube](https://github.com/pglass/cube)

- **Ce que c'est** : les bibliothèques historiques de manipulation (représentation par cubies, affichage, notation standard complète, quelques heuristiques pour pglass).
- **Pros** : notation standard et entreprises de rendu matures ; très utilisées comme vérificateurs d'états.
- **Cons** : lentes (objets Python par cubie, aucun batch), pas pensées ML, 3×3 seulement pour pycuber.
- **Vs future Rubik-Tensor** : pas concurrentes sur aucun axe ML ; à utiliser comme ou l'oracle de vérification tierce des tests d'intégration (indépendance du vérificateur).

### 7. [dwalton76/rubiks-cube-NxNxN-solver](https://github.com/dwalton76/rubiks-cube-NxNxN-solver) et [kociemba](https://github.com/muodov/kociemba)

- **Ce que c'est** : solveurs classiques de référence. dwalton76 : NxNxN, utilise des tables de lookup ; kociemba : 2-phase 3×3, quasi optimal en quelques millisecondes.
- **Pros** : solutions quasi-optimales rapides, standards de facto du cubing ; parfaits comme professeur pour l'imitation learning (comme dans le modèle HF 2×2).
- **Cons** : ce sont des boîtes noires de bout en bout, pas des environnements ; pas d'accès aux états intermédiaires performants ; Wrappers Python mais CPU-bound et mono-cube.
- **Vs future Rubik-Tensor** : pas des concurrents, des dépendances — prévus comme solvers externes dans la Feature 3.

### 8. [wata-orz/santa2023_permutation_puzzle](https://github.com/wata-orz/santa2023_permutation_puzzle) et les solutions Kaggle Santa 2023

- **Ce que c'est** : le corpus de code du challenge Kaggle Santa 2023 — branch & bound, iterative deepening A*, mémoïsation sur états symétriques, optimisation de chemins sur cube 4×4/5×5 (puzzle simplifié par permutations de symboles).
- **Pros** : le meilleur code non-ML existant sur les grands cubes ; références de performance à battre ; techniques de symétrie directement liées à la Feature 7.
- **Cons** : code de compétition (mono-fichier, non packagé), problème spécifique du challenge (permutation puzzle, rubric simplifié), aucune composante ML.
- **Vs future Rubik-Tensor** : terrain différent (recherche exacte vs modélisation ML) ; à citer comme référence de l'état de l'art non-ML que les méthodes ML cherchent ou non à égaler.

## Tableau comparatif

| Critère | DeepCubeA | EfficientCube | CayleyPy | CubeGPT-3 | magiccube | kociemba/dwalton | **Rubik-Tensor futur** |
|---|---|---|---|---|---|---|---|
| Environnement tensorisé | ✗ (C++ custom) | ✗ (numpy) | ✓ | Partiel | ✗ | ✗ | **✓ natif** |
| Batch GPU de moves | Partiel (recherche) | ✗ | ✓ | Partiel | ✗ | ✗ | **✓ (Feature 1)** |
| Tailles N×N×N génériques | ✗ | ✗ | ✓ (Cayley) | ✗ | ✓ | ✓ (solveur) | **✓** |
| Notation standard U/R/F... | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | **✓ (Feature 2)** |
| Oracle distance/solution | Partiel (3×3 dataset) | ✗ | ✓ (diffusion) | ✗ | ✓ (limited) | **✓ (exact)** | **✓ (Feature 3)** |
| Génération de données stratifiée | ✓ (figée) | ✗ | ✓ | ✓ | ✗ | ✗ | **✓ (Feature 4)** |
| Moteur de rollout autoregressif | ✗ | ✗ | ✗ | Partiel | ✗ | ✗ | **✓ (Feature 5)** |
| Protocoles SFT/DAgger/GRPO | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (Feature 6)** |
| Canonisation symétries | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ (Feature 7) |
| Qualité logicielle (CI, tests, packaging) | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | **✓ (déjà en place)** |

Lecture : les solveurs classiques dominent sur l'exactitude de la recherche, CayleyPy domine la recherche heuristique ML, mais l'intersection « environnement tensorisé batché + notation standard + oracles + protocoles séquentiels » est vide hors Rubik-Tensor futur.

## Analyse honnête des points faibles résiduels

- **Recherche heuristique** : le futur Rubik-Tensor n'embarque pas beam search / A* ; l'expérience du plan (méta-tokens) n'en dépend pas pour sa question centrale, mais un benchmark CayleyPy-like (distance apprise + beam) ne sera pas réalisable sans travail supplémentaire. Option : l'ajouter à la roadmap post-expérience.
- **Preuve empirique** : l'argument « state of the art » n'est acquis qu'à la livraison : la validation passe par la réplication des chiffres d'EfficientCube et la construction d'un première résultat original (méta-tokens), plutôt que par la seule comparaison de features.
- **Communauté** : CayleyPy a ~460 stars et des chercheurs actifs ; Rubik-Tensor démarre sans audience. Le positionnement « framework pour LLMs sur cube » est distinct, mais l'adoption dépendra de la publication de l'expérience et de la facilité de prise en main (documentation, exemples).

## Notes sur les sources

| Source | Crédibilité | Dernière mise à jour |
|---|---|---|
| [CayleyPy / cayleypy-cube (GitHub)](https://github.com/cayleypy/cayleypy) | 5/5 | 2025-2026 (actif) |
| [DeepCubeA (GitHub)](https://github.com/forestagostinelli/DeepCubeA) | 4/5 | 2019 (non maintenu) |
| [EfficientCube (GitHub)](https://github.com/kyo-takano/EfficientCube) | 4/5 | 2023 |
| [CubeGPT-3 (GitHub)](https://github.com/stanleytheli/cubegpt-3) | 3/5 | 2025 |
| [magiccube (GitHub)](https://github.com/trincaog/magiccube) | 3/5 | 2024 |
| [wata-orz/santa2023_permutation_puzzle (GitHub)](https://github.com/wata-orz/santa2023_permutation_puzzle) | 4/5 | 2023-2024 |
| [dwalton76/rubiks-cube-NxNxN-solver (GitHub)](https://github.com/dwalton76/rubiks-cube-NxNxN-solver) | 4/5 | - |
| [kociemba (muodov, GitHub)](https://github.com/muodov/kociemba) | 4/5 | - |

Réserves : les stars et états de maintenance évoluent ; les jugements sur la qualité logicielle de CubeGPT-3 et de magiccube reposent sur la lecture du code de repos communautaires sans revue externe ; les italiques de performance (« le plus rapide », « exact ») sont des affirmations d'ordre d'idée, à confronter aux benchmarks de chaque feature du cahier des charges.

## Conclusion et recommandations

1. Le positionnement est **crédible et unique** : après les ajouts du cahier des charges, Rubik-Tensor couvre une intersection vide (environnement tensorisé batché + notation standard + oracles + protocoles séquentiels LLM), que ni CayleyPy ni DeepCubeA/EfficientCube ni CubeGPT ne couvrent.
2. Il est **inatteignable de revendiquer le SOTA global** face à CayleyPy sur la recherche heuristique — le SOTA revendiqué doit être ciblé : « framework de modélisation ML pour cubes N×N×N orienté modèles séquentiels/LLM », et documenté comme tel.
3. La revendication doit être **prouvée empiriquement** : inclure dans la roadmap (a) la réplication des chiffres d'EfficientCube sur 2×2/3×3 avec les mêmes datasets, (b) la publication du résultat méta-tokens, (c) des benchmarks d'environnement (moves/s GPU, couverture BFS) comparés publiquement aux alternatives.
4. Les bibliothèques « concurrentes » de manipulation (magiccube, pycuber) et les solveurs (kociemba, dwalton76) peuvent être traités comme **dépendances et vérificateurs tiers** dans les tests, plutôt que comme rivaux.
5. 
