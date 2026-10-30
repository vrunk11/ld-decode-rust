# Rapport de performance — `feat/multi-source-stacking`

Rapport destiné à la session qui a écrit le code de la branche. Il couvre l'analyse faite sur `124e579` (« stack: lighter, parallel pass 0 with identical line records »). Aucune modification n'a été poussée sur `feat/multi-source-stacking` ; tout ce qui est décrit comme patch ou instrumentation a été fait dans un worktree local jetable.

## 1. Question posée

Pourquoi un run empilé (`-i a -i b ... -o out`) est-il beaucoup plus lent qu'un décodage normal ? Pistes demandées : trouver une façon plus efficace d'obtenir et d'appliquer la courbe de correction TBC sans tout démoduler.

## 2. Ce que fait le code (relu, branche `124e579`)

1. **Passe 0** (`crates/ld-decode-cli/src/main.rs`, `read_tbc_source`, l. 507 ; l'appel clé est l. 533) : chaque entrée est décodée une fois avec `Decoder::set_locate_only` pour garder les positions de lignes et les codes VBI (`stack.rs`, `Pass0Collector`). Jusqu'à `PASS0_JOBS = 6` entrées en parallèle (l. 570).
2. **Chronologie** (`stack.rs`, `Timeline::build`) : les champs des entrées sont appariés à la primaire par (code de frame VBI, parité). Les lignes de la primaire définissent une chronologie de lignes de longueur nominale.
3. **Flux virtuel** (`stack.rs`, `VirtualSource`, `compute_task` l. 578) : pour chaque échantillon virtuel, chaque entrée est lue à sa position (FIR sinc de 32 taps réels + FIR analytique de 128 taps complexes), verrouillage de phase sur la primaire, somme pondérée 1/σ².
4. **Passe finale** : le décodeur inchangé démodule ce flux virtuel une deuxième fois et écrit les sorties.

Donc un run à N entrées fait **N démodulations (passe 0) + 1 démodulation (passe finale) + la lecture FIR**, contre N démodulations pour N décodages séparés.

## 3. Mesures

Machine d'essai : conteneur à **4 cœurs** (Xeon 2,8 GHz, AVX2/AVX-512), `-t 4`, nightly-2026-08-29, build `--release` avec `.cargo/config.toml` (`x86-64-v3`). Fichier : `RF_Reference_40mhz_protected_cut.ddd.s16` du ticket decode-orc #330 (s16, 40 MHz, NTSC, 19 champs), utilisé **plusieurs fois comme entrées** (les lignes sont donc identiques d'une entrée à l'autre ; ça mesure le coût, pas la qualité). Ce n'est **pas** ton setup (6 captures à 75 MHz, 32 threads) : seuls les rapports sont transposables.

| Run | Temps |
| --- | --- |
| 1 entrée (`-i f -o o`, mode resample), `-l 2` | 2,0 s |
| 3 entrées, `-l 2` | 7,3 s |
| 3 entrées, sans `-l` | 8,0 s |
| 1 entrée, sans `-l` | 5,5 s |
| 2 entrées / 3 entrées, sans `-l` | 7,3 s / 8,4 s |
| 3 entrées : passe 0 / passe finale | 5,5 s / 3,8 s |
| Lecture du flux virtuel, 3 entrées (`LD_TIMING=1`, 13 lectures) | 12,7 M échantillons en 2,7 s, soit 4,7 M éch/s |

Répartition dans `compute_task` (instrumentation locale `LD_TASKT`, 199 tâches, CPU-secondes cumulées sur les threads, 3 entrées) :

| Phase | Temps | Part |
| --- | --- | --- |
| `plan` (positions, lecture fichier) | 0,12 s | 2 % |
| FIR (sinc 32 + analytique 128) | 6,63 s | 91 % |
| Verrouillage de phase et somme | 0,55 s | 8 % |

(Les durées de phase incluent l'attente du parallélisme imbriqué, donc la répartition est plus fiable que les valeurs absolues.)

## 4. Constats

### 4.1 La passe 0 ignore `-s` et `-l` (confirmé)

`read_tbc_source` appelle `decode_all(..., None, 0, ...)` : pas de limite de frames, début à 0. Le temps de la passe 0 est le même avec `-l 2` qu'avec toute la capture (19 champs traités dans les deux cas ; 1 entrée seule passe de 5,5 s à 2,0 s avec `-l 2`, 3 entrées de 8,0 s à 7,3 s).

Conséquence : un test de 30 frames sur des captures de 10 à 14 s traite 10 à 14 s de vidéo par entrée. Ton commit donne 109 s (une entrée à la fois) / 98 s (six à la fois) pour un résultat de 30 frames. **Je n'ai pas pu vérifier sur 75 MHz que c'est la totalité de ton 1m38** ; c'est cohérent avec tes chiffres, c'est une déduction.

### 4.2 La boucle FIR de `compute_task` est lente

Elle fait 288 multiplications-accumulations par échantillon d'entrée (32 réelles + 128 complexes × 2) et tourne à ~190 ns par échantillon d'entrée, environ 10× sous le pic AVX2. Le build est bien en `x86-64-v3` ; ce n'est pas un problème de drapeaux.

- **Patch testé dans le worktree** : accumulateurs multiples (4 pour le sinc, 8 pour la bande analytique, en f64). Gain mesuré : lecture de 4,7 à ~6,3 M éch/s (**+35 %**), sortie `.tbc` identique à la version d'origine. L'accumulation en série n'est donc pas la cause principale.
- **Micro-benchmark isolé** (`scripts/stacking_perf/fir_bench.rs`, `rustc -O -C target-cpu=x86-64-v3 fir_bench.rs`) qui reproduit ces boucles sur 1 M d'échantillons : version de la branche (accumulation série en f64, table complexe entrelacée) **191 ns** par échantillon d'entrée, concordant avec les ~174 ns déduits du vrai run ; version à tables re/im séparées en f32, 8 voies, **45 ns** (**4,2×**). Écart relatif max 3,15e-4 sur mes données de test, qui ont de fortes annulations ; l'effet sur du vrai signal est **non vérifié**. Ton backend OpenCL calcule déjà les produits scalaires en simple précision et annonce un résultat identique « en bruit ».
- Cette version à 45 ns reste à ~2,3 multiplications-accumulations par cycle, donc loin du pic : il y a encore de la marge (FMA explicite via `mul_add`, tables plus compactes).

### 4.3 La passe finale démodule une deuxième fois

Inhérent à la conception actuelle (le décodeur inchangé tourne sur le flux virtuel). Aucune des pistes ci-dessous ne l'enlève.

### 4.4 Observation non expliquée

Le run empilé produit **16 champs** là où une entrée seule produit **19**, sur le même fichier. La chronologie couvre 4998 lignes (0,32 s) pour un fichier de 0,367 s. Je n'ai pas cherché la cause ; probablement un effet de bord au début ou à la fin de la chronologie, à vérifier.

## 5. Pistes (rien d'implémenté sur la branche)

Classées de la plus sûre à la plus risquée.

1. **Restreindre la passe 0 à la fenêtre `-s` / `-l` de la primaire, plus une marge, et retrouver pour chaque entrée la frame correspondante par le VBI** (la logique de `seek_getframenr` / `-S` existe déjà). Enregistrements identiques bit à bit à une passe complète sur la même fenêtre.
2. **Mettre en cache les enregistrements de la passe 0** (`FieldRec` : parité, code VBI, positions des lignes) par fichier, avec taille et date de modification comme clé. Utile pour toutes les expériences répétées sur les mêmes captures.
3. **Réécrire le FIR** avec des tables re/im séparées en f32 (voir 4.2). À valider sur du vrai signal contre la version f64 ; ton script de bruit sur les lignes noires (0,749 / 0,682 IRE) est le bon critère.
4. **Réduire le travail du FIR** : calculer le signal analytique de chaque entrée une fois par FFT sur chaque étendue lue, puis le lire avec un sinc complexe court (~64 multiplications-accumulations au lieu de 288). Plus de travail et de validation (en particulier la fuite au croisement de 3 MHz, aujourd'hui à −37 dB).
5. **Passe 0 allégée pour les secondaires seulement.** Seule la primaire garde la démod complète. Les secondaires n'ont besoin que d'un délai relatif, que la méthode de la section 4 bis de `stacking-handoff.md` obtient sans le noyau complet (signal passe-bas, front de chaque ligne, corrélation ±512). Un démodulateur léger de type « tranche du spectre autour de la porteuse + FFT inverse courte » (comme l'étage 1 de l'audio dans `demodblock.rs`) serait à écrire. **Coût et précision non mesurés.** Contrainte : garder la courbe relative sous ~0,25 échantillon (mon spike : à 35 dB de CNR, 0,5 échantillon d'erreur perd ~70 % du gain d'empilement).
6. **Lire le VBI sans démoduler toute la capture** (condition de la piste 5) : seules quelques lignes par champ portent le VBI, mais la démod travaille par blocs de 32 768 échantillons. Démoduler seulement les blocs autour des lignes VBI une fois la position approximative connue. À concevoir ; conflit possible avec le MTF (la passe 0 actuelle a besoin de la démod complète parce que la calibration MTF lit les niveaux RF).

Recommandation : 1 et 2 d'abord (sûres, mesurables, gros effet sur les tests courts), puis 3 ; les pistes 5 et 6 demandent un prototype mesuré sur tes six captures.

## 6. Ce qui n'a pas été vérifié

- Aucune mesure sur 75 MHz, 6 entrées, 32 threads ou GPU. Les chiffres du conteneur (4 cœurs) ne sont valables que comme rapports.
- Entrées identiques (même fichier trois fois) : les lignes concordent exactement, donc aucun saut ni écart n'est exercé et aucune garde d'alignement.
- Le gain f32 du micro-benchmark vient de données synthétiques, hors qualité d'image.
- Les patchs et l'instrumentation du worktree ne sont pas dans le dépôt (seul `fir_bench.rs` y est).

## 7. Reproduire

```bash
git fetch origin feat/multi-source-stacking
RUSTUP_TOOLCHAIN=nightly-2026-08-29 cargo build --release
B=target/release/ld-decode; F=<capture>.s16
# passe 0 ignore -l : comparer
$B -t 4 --format s16 -i $F -o o1 -l 2
$B -t 4 --format s16 -i $F -i $F -i $F -o o3 -l 2
# durée des lectures du flux virtuel
LD_TIMING=1 $B -t 4 --format s16 -i $F -i $F -i $F -o o3 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep '^STACK read'
# micro-benchmark du FIR
rustc -O -C target-cpu=x86-64-v3 scripts/stacking_perf/fir_bench.rs -o fir_bench && ./fir_bench
```
