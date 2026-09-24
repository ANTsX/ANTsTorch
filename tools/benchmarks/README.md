# Catalogue des modèles — `tools/benchmarks/`

Ce fichier documente, à un seul endroit, tous les noms de modèles acceptés
par les scripts de comparaison sous `tools/benchmarks/` et par
`antstorch.benchmark.evaluate_mindboggle_pair()` qu'ils appellent. Chaque
entrée renvoie vers la docstring qui fait réellement autorité (celle-ci est
un résumé de navigation, pas une nouvelle source de vérité — en cas de
divergence, la docstring du code a toujours raison).

## Scripts et ce qu'ils comparent

| Script | Compare | Modèles/`--models` |
|---|---|---|
| `run_benchmark_mindboggle_probe.py` | Tous les modèles natifs ANTsTorch entre eux, sur un petit jeu de paires Mindboggle-101 (démo/pilote local, sans CLI d'orchestration résumable — voir sa propre docstring) | Les 6 variantes natives par défaut, voir tableau ci-dessous ; `--models` accepte en fait *tout* nom que `evaluate_mindboggle_pair()` reconnaît, y compris les quatre `*_regadam`, tout `_ANTS_TRADITIONAL_MODELS`, ou tout `type_of_transform` ANTs déformable-seul passé tel quel |
| `compare_syntx_antstorch_mindboggle.py` | syntx vs antstorch, famille SyN dense | `gaussian`, `sobolev`, `dsti`, `bspline`, `gaussian_regadam`, `sobolev_regadam`, `dsti_regadam`, `bspline_regadam` |
| `compare_fireants_mindboggle.py` | syntx vs antstorch vs FireANTs (3ᵉ bras) | Même liste que ci-dessus, mais FireANTs n'est réellement évalué que pour `gaussian` (les autres modèles restent affichés « n/a » côté FireANTs pour garder les colonnes syntx/antstorch renseignées) |
| `compare_bspline_scattered_data.py` | ANTs vs ANTsTorch, ajustement B-spline sur données éparses (pas un modèle de recalage) | N/A — comparaison de primitive, pas de `--models` |
| `compare_n4_bias_field_correction.py` | ANTs vs ANTsTorch, correction de champ de biais N4 (pas un modèle de recalage) | N/A — comparaison de primitive, pas de `--models` |
| `compare_denoise_image.py` | ANTs `denoise_image` vs `antstorch.denoise_image` (moyennes non locales adaptatives, bruit ricien ou gaussien) | N/A — comparaison de primitive, pas de `--models` ; `--repeats 2+` rapporte aussi la variabilité d'une exécution ANTs à l'autre |
| `compare_cortical_thickness.py` | ANTs `kelly_kapowski` vs moteur DiReCT tensoriel ANTsTorch sur une segmentation Deep Atropos partagée (`S_template3` par défaut) | N/A — mêmes entrées GM/WM et mêmes paramètres partagés ; `--optimizer` et `--regularizer` sélectionnent la variante ANTsTorch |
| `run_ants_comparisons.sh` | Lance à la suite `compare_denoise_image.py`, `compare_n4_bias_field_correction.py` et `compare_cortical_thickness.py` sur un même périphérique (`DEVICE=cuda:0` par défaut), avec un journal par comparaison | N/A — variables `DEVICE`, `OUT_DIR`, `PYTHON`, `REPEATS`, `ANTS_THREADS` |

## Modèles natifs ANTsTorch (`antstorch.benchmark.evaluate_mindboggle_pair(model=...)`)

Documentation faisant autorité : docstring de
`antstorch.benchmark.evaluate.evaluate_mindboggle_pair()` (mise à jour à
chaque nouveau bras) et, pour les détails de recalage eux-mêmes, docstring
de `antstorch.syn.syn_registration()`.

### Famille SyN dense (`antstorch.syn.syn_registration(type_of_transform="SyNOnly", regularizer=..., ...)`)

Les quatre variantes `*_syn` ne diffèrent que par le régularisateur fluide
appliqué à chaque itération ; toutes partagent la même affine canonique par
paire (invariant d'équité, projet §17/§18).

| Nom de modèle | Régularisateur | Optimiseur | Remarque |
|---|---|---|---|
| `gaussian_syn` | `gaussian` (lissage gaussien séparable) | `gradient_descent` (défaut) | — |
| `sobolev_syn` | `sobolev` (opérateur de Green spectral) | `gradient_descent` (défaut) | — |
| `dsti_syn` | `dsti` (opérateur de Green DST-I, Dirichlet homogène) | `gradient_descent` (défaut) | Écart persistant vs le bras `dsti` de syntx, documenté §34/§36 du projet |
| `bspline_syn` | `bspline` (ANTs/ITK `BSplineSyN`, ajustement B-spline cubique par niveau) | `gradient_descent` (défaut) | Bras non-officiel côté syntx (voir plus bas) |
| `gaussian_regadam` | `gaussian` (même régularisateur que `gaussian_syn`) | **`reg_adam`** | Teste l'effet de l'optimiseur seul sur `gaussian`, sans toucher à `gaussian_syn` |
| `sobolev_regadam` | `sobolev` (même régularisateur que `sobolev_syn`) | **`reg_adam`** | Idem pour `sobolev` |
| **`dsti_regadam`** | `dsti` (même régularisateur que `dsti_syn`) | **`reg_adam`** | Premier bras `_regadam` créé — teste si l'optimiseur seul explique une partie de l'écart `dsti` vs syntx, sans toucher à `dsti_syn` (projet §40) |
| `bspline_regadam` | `bspline` (même régularisateur que `bspline_syn`) | **`reg_adam`** | Idem pour `bspline` (bras non-officiel côté syntx également, voir plus bas) |

Les quatre bras `*_regadam` ci-dessus utilisent un portage léger de
l'optimiseur Adam-momentum de `syntx/greedy.py` (`syn_registration(...,
optimizer="reg_adam")`, projet §40/§41). `optimizer` et `regularizer` sont
deux paramètres indépendants de `syn_registration()` — `reg_adam` fonctionne
avec n'importe lequel des quatre régularisateurs, pas seulement `dsti` (où
il a été introduit en premier). Aucun bras `*_syn` correspondant n'est
modifié — chaque `*_regadam` est un bras entièrement séparé, préservant la
reproductibilité de tout run antérieur. Pas encore exécutés sur données
Mindboggle-101 réelles — seulement validés sur le jeu de test synthétique.

`gaussian_sigma_mode`/`conservative_smooth` (kwargs optionnels) permettent de
reproduire les conventions par défaut de `syntx.syn` pour `gaussian`/
`sobolev`/`dsti` (`*_syn` et `*_regadam` indifféremment) plutôt que celles,
différentes, d'ANTsTorch — voir la docstring de `syn_registration()` pour le
détail exact de ce qui change. `adam_betas`/`adam_eps` (kwargs optionnels,
uniquement pertinents pour les bras `*_regadam`) contrôlent la décroissance
des moments Adam.

### Famille champ de vitesse stationnaire (SVF, pas SyN)

| Nom de modèle | Backend | Remarque |
|---|---|---|
| `bspline_svf` (alias `svf`) | `antstorch.bspline_flows.bspline_svf_registration()` | Paramétrisation B-spline du SVF — partage le mot « bspline » avec `bspline_syn` mais c'est une famille de transformation différente (SVF, pas SyN) |
| `gaussian_svf` | `antstorch.bspline_flows.gaussian_svf_registration()` | SVF dense (voxel par voxel) à lissage gaussien update/total-field |

### Baselines ANTs traditionnels (non-ANTsTorch)

| Nom de modèle | Dispatch |
|---|---|
| `ants_syn_quick` (clé de `_ANTS_TRADITIONAL_MODELS`) | `ants.registration(type_of_transform="antsRegistrationSyNQuick[so]")` direct, sur la même affine canonique partagée |
| tout `type_of_transform` `ants.registration()` **déformable-seul** (ex. `"antsRegistrationSyNQuick[so]"`, `"SyNOnly"`, `"antsRegistrationSyN[bo]"`) | Passé tel quel, sensible à la casse — un preset non déformable-seul (ex. `"...[s]"`) est rejeté explicitement, car il recalculerait sa propre affine et casserait l'invariant d'équité |

## Modèles côté comparaison syntx / FireANTs

Documentation faisant autorité : docstring en tête de
`compare_syntx_antstorch_mindboggle.py` et de `compare_fireants_mindboggle.py`.

| Nom `--models` | Bras syntx | Bras antstorch | Bras FireANTs (script fireants seulement) |
|---|---|---|---|
| `gaussian` | `syntx.benchmark.evaluate_mindboggle_pair(model="gaussian")` | `gaussian_syn` | `fireants.registration.greedy.GreedyRegistration` (seul régularisateur supporté) |
| `sobolev` | `model="sobolev"` | `sobolev_syn` | n/a |
| `dsti` | `model="syn_dsti1"` (bras `dsti1`/`reg_adam` propre à syntx) | `dsti_syn` | n/a |
| `bspline` | Bras **non-officiel** : `syntx.syn(regularizer="bspline", ...)` appelé directement en mimant la branche `sobolev` du harnais syntx — syntx n'expose pas ce nom de modèle dans son propre `evaluate_mindboggle_pair()` | `bspline_syn` | n/a |
| `gaussian_regadam` | `model="gaussian"` — **même bras syntx que `gaussian`**, pas de contrepartie séparée | `gaussian_regadam` | n/a |
| `sobolev_regadam` | `model="sobolev"` — **même bras syntx que `sobolev`** | `sobolev_regadam` | n/a |
| `dsti_regadam` | `model="syn_dsti1"` — **même bras syntx que `dsti`** | `dsti_regadam` | n/a |
| `bspline_regadam` | Même bras non-officiel que `bspline` — **même bras syntx que `bspline`** | `bspline_regadam` | n/a |

Aucun des quatre `*_regadam` n'a de contrepartie syntx propre : chacun est
comparé au même bras syntx que son homologue de base (voir `_base_reg()`
dans le script). `--matched` force `grad_step`/métrique de similarité/
schedule de pyramide identiques des deux côtés pour `gaussian`/`sobolev`/
`bspline` (les trois variantes à descente de gradient simple) ; `dsti` et
les quatre `*_regadam` ne reçoivent que l'alignement de la formule du
régularisateur (`conservative_smooth=True` pour sobolev/dsti,
`gaussian_sigma_mode="voxel"` pour gaussian, rien pour bspline), pas
l'alignement de schedule — puisque l'optimiseur syntx (`reg_adam`/TVF)
diffère fondamentalement pour `dsti`, et que l'optimiseur `reg_adam` est
précisément la variable testée pour les quatre `*_regadam`, forcer le reste
n'isolerait rien. Voir projet §34 pour le détail de cette décision, et
§40/§41 pour les bras `*_regadam` spécifiquement.

## Historique

Les décisions de conception derrière ce catalogue (pourquoi tel bras existe,
pourquoi tel autre est resté hors-périmètre) sont documentées en détail dans
le document du projet Claude « Cadre du recalage d'ANTsTorch »
(`claude/syntx-antstorch-integration-proposal.md`), en particulier :
- §17-19 : convention de nommage `_syn`, invariant d'affine canonique partagée.
- §34/§36 : diagnostic de l'écart `dsti_syn` vs `dsti` (syntx), et sa
  variance run-à-run.
- §37-39 : intégration de FireANTs comme troisième bras.
- §40 : portage `RegAdam`/`dsti_regadam`.
- §41 : généralisation de `RegAdam` aux trois autres régularisateurs
  (`gaussian_regadam`, `sobolev_regadam`, `bspline_regadam`).
