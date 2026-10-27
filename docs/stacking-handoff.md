# Stacking multi-source — document de reprise

Ce document résume la session de conception et de spike pour la reprendre ailleurs. La conception détaillée est dans [`stacking-design.md`](stacking-design.md) ; ici : où on en est, ce qui est établi, ce qui ne l'est pas, et quoi faire ensuite.

Branche : `ccr-74cb4fe9-ehem1s` (rien n'est fusionné dans `main`). **Aucun code Rust du décodeur n'a été modifié.** Tout ce qui existe est de la documentation et un prototype Python dans `scripts/stacking_spike/`.

## 1. Objectif

Aujourd'hui on démodule chaque capture RF d'un même LaserDisc en `.tbc`, puis on empile les `.tbc`. Ça marche mais ce n'est pas optimal : dropouts, erreurs de sync et bruit sont déjà figés, et l'information de fiabilité du RF est perdue. Empiler le RF directement est trop lourd et exige un alignement de phase de porteuse irréaliste. Objectif : **une démodulation qui bénéficie de toutes les sources**, avec N sources quelconques.

## 2. Décisions prises avec toi

- **N quelconque**, agnostique de la carte de capture et du lecteur (la fréquence instantanée est une grandeur physique du disque ; chaque source garde son `DecoderSpec`).
- **Toutes les sources contribuent**, la primaire comprise. La primaire ne fournit que la grille temporelle (plus l'EFM et l'audio tant que leur phase n'est pas faite). Choix automatique par meilleur SNR, avec forçage possible.
- **Estimation conjointe, par accumulation de preuves** : pour chaque échantillon, on cherche la fréquence la plus probable sachant toutes les sources, avec des poids continus. Pas de choix binaire entre sources ; seul `g_i = 0` reste binaire (hors couverture, skip de disque, dropout franc).
- **L'alignement est dynamique** : le décalage est une adresse de lecture qui varie avec la position (offset, vitesse de lecture relative, oscillation du disque, jitter). On ne rééchantillonne pas le RF.
- **Hypothèse** : aucun échantillon perdu dans une capture ; les seules ruptures sont des **skips de disque**.
- **Vérification VBI source par source et frame par frame** avant d'accepter une source sur une frame (via `Decoder::scan_field`, qui renvoie déjà `frame_number`).
- **Ordre** : vidéo d'abord, puis audio analogique (même estimateur sur le signal décimé), puis EFM (accumulation de formes d'onde avec égalisation par source).
- **Parité** : chemin multi-source opt-in (`--stack <fichier>`, répétable) ; avec une seule entrée, le code actuel est inchangé et les sorties restent byte-identiques à la référence Python.

## 3. Architecture proposée (résumé)

`demod_block_cpu` (`crates/ld-decode/src/decode/demodblock.rs`) serait découpé en : **étage A par source** (FFT, filtre RFVideo, signal analytique), **estimateur conjoint** (remplace `unwrap_hilbert_into` ; un seul `atan2` par échantillon), **étage B une seule fois** (clip, filtres vidéo, de-emphasis, `demod_05`, `demod_burst`, inchangé). Une **passe d'alignement** séquentielle précède la démodulation parallèle et produit une table de délais τ_S(position) par source, lue en lecture seule (le décodeur démodule en parallèle et peut revenir en arrière avec `redo`).

Estimateur : pour la source i, `p_i = z_i(k)·conj(z_i(k−1))` (invariant à la phase de porteuse), poids de fiabilité `g_i` (≈ Â²/σ², réduit quand l'enveloppe instantanée plonge), `Δ̂ = arg Σ g_i·p̂_i`, repondération robuste douce, puis f̂ = Δ̂·fs/2π. Voir `stacking-design.md` pour le détail.

## 4. Ce que le spike a établi

Prototype numpy, **pas du Rust**, signal synthétique puis un seul vrai fichier. Détails et tableaux dans `stacking-design.md`.

- Avec un alignement parfait, le gain est le gain théorique : −3,0 dB à 2 sources, −4,8 dB à 3 (synthétique et signal réel).
- **L'alignement est le facteur limitant.** À fort CNR (35 dB), 0,5 échantillon d'erreur perd ~70 % du gain et 1 échantillon est pire que sans stacking. Il faut viser ≤ 0,25 échantillon.
- **Le mode entier ne suffit pas sur du vrai contenu** : il peut être pire que la source seule (4,13 contre 3,99 IRE). Le mode fractionnaire est nécessaire.
- **Le suiveur de délai actuel** (corrélation croisée normalisée sur le signal démodulé passe-bas, 16384 puis 2048 échantillons, lissage) donne 0,25 à 0,3 échantillon de résiduel à fort CNR avec un jitter lent, 0,64 à CNR bas, et **1,4 avec un jitter rapide, cas où le résultat empilé est bien pire que la source seule** (8,26 contre 3,94 IRE).
- **L'estimateur conjoint ne bat pas une moyenne pondérée par 1/σ²** en RMS (écart ≤ 1 %, souvent à l'avantage de la moyenne). Un seul avantage observé : moins de clics FM résiduels à 12 dB avec 2 sources (10-16 contre 44-59), non reproduit à 3 sources.
- Le retard de groupe sur le spectre croisé des signaux analytiques a été essayé et **ne marche pas** (erreur de 0,7 à 1,3 échantillon même en test propre).
- La correction de cadence locale (différence de vitesse de lecture) apporte peu dans le modèle testé.

## 5. Ce qui n'est PAS établi

- Aucune **vraie paire de captures** n'a été testée : le seul fichier réel (ticket decode-orc #330) a servi à fabriquer des copies déformées par mes modèles de time-base, dont le bruit d'origine est commun. Le jitter réel entre deux lecteurs est inconnu.
- Un **seul tirage** par cas, bruit gaussien blanc, modèles de jitter inventés (lent : corrélé sur ~50 µs ; rapide : ~8 µs).
- La forme des poids `g_i` est une approximation non optimisée.
- Le coût réel en Rust (passe d'alignement, N× l'étage A) n'est pas mesuré.
- Le ré-encodage d'un `.tbc` en FM avec time-base déformée n'est pas fait (il demande une pre-emphasis cohérente avec la de-emphasis du décodeur).
- La **combinaison cohérente avant discriminateur** (MRC) n'a pas été testée : tu avais écarté « combiner », mais c'est la seule piste qui améliore la porteuse avant la non-linéarité et pourrait gagner près du seuil FM.

## 6. Reproduire

Prérequis : `python3` avec `numpy` (pas de scipy). Les scripts importent `stack_spike` depuis leur dossier : lancer depuis `scripts/stacking_spike/`.

```bash
cd scripts/stacking_spike
# synthétique : CNR par source, corrélation du jitter (échantillons), graine
python3 sweep.py 25,25 2000 1
python3 sweep.py 35,35 2000 1
# signal réel : fichier s16 40 MHz, CNR du bruit AJOUTÉ par source, jitter, graine, début
python3 real_spike.py /chemin/capture.s16 30,30 2000 1 [début_en_échantillons]
```

Chaque exécution prend quelques minutes (2 M d'échantillons). Le décodeur Rust demande le nightly de la README : `rustup toolchain install nightly-2026-08-29` puis `RUSTUP_TOOLCHAIN=nightly-2026-08-29 cargo build --release` (3 min 20 s dans le conteneur ; `clang++` requis). Le fichier du ticket n'est pas dans le dépôt.

## 7. Suite recommandée (par priorité)

1. **Avec tes vraies captures** : mesurer le résiduel d'alignement sur de vraies paires. Sans vérité, utiliser la méthode des trois captures : pour trois captures indépendantes a, b, c recalées, `var(a−b) = σa² + σb²` (idem pour les deux autres paires), ce qui donne le bruit de chacune et celui du stack sans référence. Les erreurs d'alignement gonflent ces différences, donc elles se mesurent aussi.
2. **Alignement fin** : par les fronts de sync de chaque source (positions des lignes avec interpolation sous-échantillon, comme les linelocs du décodeur), ou corrélation sur un signal moins filtré. Cible ≤ 0,25 échantillon, y compris avec un jitter rapide.
3. **Garde de confiance** : pondérer chaque source par la qualité de son alignement local (par exemple la corrélation entre la primaire et la source recalée) pour qu'un alignement douteux ramène son poids vers 0. Indispensable : sans elle, un jitter rapide dégrade l'image.
4. **Mode fractionnaire** d'office.
5. **Tester la combinaison cohérente avant discriminateur** si tu l'acceptes, seulement près du seuil FM.
6. Ensuite seulement : refactor Rust (étage A / estimateur / étage B avec parité N=1), vérification VBI, table de délais, option CLI, audio puis EFM.

## 8. Fichiers

- `docs/stacking-design.md` : conception complète et résultats détaillés.
- `docs/stacking-handoff.md` : ce document.
- `scripts/stacking_spike/stack_spike.py` : générateur synthétique, suiveur de délai, estimateurs, scoring.
- `scripts/stacking_spike/sweep.py` : balayage synthétique avec erreur d'alignement injectée.
- `scripts/stacking_spike/real_spike.py` : même test sur un vrai fichier s16.
