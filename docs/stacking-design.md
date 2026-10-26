# Stacking multi-source dans la démodulation — conception

Statut : proposition, rien n'est implémenté. Les chiffres marqués « à valider » sont des estimations à confirmer par le spike (voir [Plan](#plan)).

## Objectif

Combiner N captures RF (N quelconque) du même disque pour obtenir une meilleure démodulation qu'un stack de `.tbc` (qui a déjà figé dropouts, erreurs de sync et bruit, et perdu l'enveloppe RF) et moins coûteuse qu'un stack RF (qui exige un alignement de phase de porteuse à une fraction d'échantillon, impossible à tenir avec la dérive de chaque lecteur).

## Principe

Combiner **après le discriminateur FM**. La fréquence instantanée (Hz) est une grandeur physique du disque : elle ne dépend pas de la phase de porteuse, ni de la fréquence d'échantillonnage, du format ou du gain de la carte. Seul un alignement temporel est nécessaire.

## Insertion dans le code

`demod_block_cpu` (`crates/ld-decode/src/decode/demodblock.rs`) est découpé en deux :

- **Étage A, par source :** `fft → hilbert → unwrap_hilbert` donne `freq[]` (Hz) et `env[]` (`|hilbert|`).
- **Combineur :** produit un `freq[]` unique sur la grille de la source primaire.
- **Étage B, une seule fois :** clip, FFT, filtres `FVideo`, de-emphasis, `demod_05`, `demod_burst` — inchangé.

Les appels à `demod_block_cpu` dans `decode/mod.rs` prennent déjà une tranche `&[f32]` ; c'est la jonction.

Chaque source est démodulée avec son propre `DecoderSpec` (filtres de sa fréquence d'échantillonnage). Le RF n'est jamais rééchantillonné ni modifié.

## Alignement

Le délai relatif τ_S(t_P) d'une source S par rapport à la primaire P est une fonction dynamique de la position. Il a trois composantes :

1. offset constant (départ des captures) ;
2. pente (horloges d'échantillonnage et vitesse moyenne de rotation différentes) ;
3. jitter de time-base fort : excentricité et oscillation du disque (revenant à chaque tour), flutter, servo. Chaque lecteur a le sien, avec sa propre phase dans le tour ; on ne suppose aucun alignement entre captures. La plage de τ peut atteindre des dizaines à des centaines d'échantillons (à valider).

Estimation, en coarse-to-fine, par source contre la primaire (topologie en étoile) :

- **Verrouillage grossier :** numéro de frame VBI (CAV/CLV) et `fileLoc` des champs.
- **Niveau 1, global :** pente et offset sur de longues fenêtres.
- **Niveau 2, ~1 ligne :** corrélation croisée par FFT sur des fenêtres de ~2000 échantillons avec recouvrement, pic interpolé au sous-échantillon, plage de recherche large.
- **Niveau 3, itératif :** on applique le warp, puis on re-corrèle sur des fenêtres plus courtes (~500).
- Lissage sous contrainte de régularité, état [τ, τ̇].

Hypothèse : aucun échantillon perdu dans une capture. La relation échantillon → temps est donc continue et monotone ; la seule cause de perte de verrouillage est un **skip de disque**.

### Application du décalage

Le décalage est une adresse de lecture dynamique : pour l'échantillon n de la primaire, on lit la secondaire en `n + τ_S(n)`.

- **Entier (défaut) :** `m = n + round(τ)`, zéro interpolation. Erreur de quantification ±0,5 échantillon (~7 ns RMS à 40 MHz).
- **Fractionnaire (si le spike montre que l'entier pénalise le SNR) :** interpolation courte (~8 coefficients) de `freq[]` seulement, jamais du RF. `freq[]` est lisse et suréchantillonné (bande vidéo ~5 MHz pour 40 MHz), donc l'interpolation ne le dégrade pas.

Critère d'acceptation : résiduel RMS d'alignement ≤ ~0,1 échantillon à 40 MHz (~2,5 ns), soit ~3° de phase sur la sous-porteuse couleur (< 0,1 % de perte de chroma).

### Skips de disque

Chaque source est découpée en segments de continuité (perte de corrélation, discontinuité du numéro de frame VBI). Au skip, la source passe à poids 0 puis est recalée sur la bonne position du disque (frame VBI, puis niveau 2). Sans VBI lisible : recherche par corrélation sur une fenêtre large, en respectant la parité de champ. Si c'est la primaire qui skippe, elle impose sa timeline comme dans un décodage normal.

## Combineur

Toutes les sources contribuent, la primaire comprise (elle ne fournit que la grille, l'EFM et l'audio).

- Poids `wᵢ = 1/σᵢ²`, σᵢ étant le bruit estimé par source et par segment (résidu hors bande vidéo de `freq[]`). Indépendant du gain, de la profondeur de bits et du format ; tient compte de la bande utile de chaque capture.
- L'enveloppe ne sert qu'au rejet des dropouts, relativement à la médiane de la source.
- Rejet des clics FM : écart à la médiane (N ≥ 3) ou à l'autre source (N = 2).
- `f = Σ wᵢ·fᵢ / Σ wᵢ`.
- Poids 0 uniquement : dropout local, skip en cours de recalage, zone non couverte.
- Calibration par source avant combinaison : régression linéaire de `f_S` contre `f_P` sur chaque segment aligné (corrige offset et gain Hz entre lecteurs).

Avec un bruit indépendant, une mauvaise source ne dégrade jamais le résultat. Gain attendu : environ 3 dB à 2 sources, 4,8 dB à 3 (bruit non corrélé). Les défauts physiques du disque présents dans toutes les captures subsistent, comme avec un stack TBC.

## Choix de la primaire

Automatique, par meilleur SNR mesuré sur un pré-passage court (quelques centaines de champs) ; `--stack-primary` pour forcer.

## Architecture d'exécution

Le décodeur démodule en parallèle et dans le désordre (prefetch), et `decode()` peut revenir en arrière (`redo`). Le délai doit donc être une fonction figée de la position :

1. **Passe d'alignement**, séquentielle et légère : calcule la table τ_S(t_P) par source (nœuds toutes les quelques centaines d'échantillons).
2. **Passe de démodulation stackée**, parallèle : lit la table en lecture seule.

Coût de la passe 1 à trancher par mesure : (A) étage A complet, ou (B) discriminateur approximatif sur signal décimé.

Mémoire en O(N) ; l'étage A coûte N×, l'étage B 1×.

## Contraintes

- **Parité préservée :** le chemin multi-source est opt-in (`--stack <fichier>`, répétable). Avec une seule entrée le code et les sorties restent byte-identiques aux références, et les tests de parité existants restent le garde-fou.
- **EFM** (dépend de la phase RF) et **audio** : pris sur la primaire dans un premier temps. L'audio analogique pourra être démodulé par source plus tard.
- **Mesure :** `VitsMetrics` existe déjà ; on compare une source seule, le stack TBC actuel et le stack en domaine démodulé.

## Plan

1. **Spike go/no-go :** paire synthétique à partir d'une capture (offset, pente ~20 ppm, wow, jitter à l'échelle de la ligne, bruit indépendant, gain et fréquence d'échantillonnage différents, un skip). Mesurer le résiduel d'alignement selon la taille de fenêtre, la variante A/B, et le mode entier/fractionnaire.
2. Refactor étage A / étage B, avec parité N=1 vérifiée.
3. Lecture multi-source et verrouillage grossier.
4. Suiveur de délai, table τ, combineur.
5. Option CLI et comparaison du SNR avec le stack TBC.
