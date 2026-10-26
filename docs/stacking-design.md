# Démodulation conjointe multi-source — conception

Statut : proposition, rien n'est implémenté. Les chiffres marqués « à valider » sont des estimations à confirmer par le spike (voir [Plan](#plan)).

## Objectif

Utiliser N captures RF (N quelconque) du même disque pour que **la démodulation elle-même** soit meilleure, au lieu d'empiler des sorties déjà démodulées (stack `.tbc` ou moyenne de fréquences en sortie de discriminateur : les clics FM, dropouts et erreurs de sync y sont déjà figés).

## Principe : estimer la fréquence f(n) sachant toutes les sources

Pour chaque échantillon n de la grille de la source primaire, on cherche la fréquence f(n) **la plus probable** (maximum de vraisemblance, avec un a priori de continuité) étant donné les observations de **toutes** les sources. Les autres sources servent de preuves pour décider ce qu'il faut démoduler, y compris aux endroits où une source seule se trompe (clics FM, enveloppe faible, dropout).

Le RF n'est jamais rééchantillonné ni modifié.

### Modèle d'observation

Pour la source i, à l'échantillon aligné k = n + m_i(n) (m_i : décalage entier dynamique, voir [Alignement](#alignement)) :

- `z_i(k)` : signal analytique (sortie du filtre de Hilbert, avec le filtre `RFVideo` et le `DecoderSpec` propres à la source).
- `p_i(n) = z_i(k) · conj(z_i(k−1))`, les deux échantillons pris au **même** offset m_i(n), donc un vrai couple consécutif de la même source. Son argument est le pas de phase Δ, donc la fréquence f = Δ·fs/2π.
- `p_i` est **invariant à la phase de porteuse** : aucun recalage de phase entre sources n'est nécessaire (contrairement à un stack RF).
- Bruit complexe gaussien de variance σ_i² (estimée par source et par segment).
- `Â_i(n)` : amplitude de porteuse attendue, estimée localement (médiane glissante). Jamais la valeur instantanée, qui contient le bruit.

### Estimateur

1. **Vraisemblance.** `L(Δ) = Σ_i g_i(n) · Re(p̂_i(n) · e^{−jΔ})`, avec `p̂_i = p_i/|p_i|` et `g_i(n)` la fiabilité locale de la source, approximativement `Â_i/σ_i²`, réduite quand l'enveloppe instantanée `|z_i|` passe sous `Â_i` (précurseur de clic). `g_i = 0` si la source est en dropout, en skip de disque ou hors couverture. La forme exacte de `g_i` est à fixer au spike.
2. **Estimation.** `Δ̂ = arg Σ_i g_i · p̂_i`, puis `f̂ = Δ̂·fs/2π` (mêmes conventions que `unwrap_hilbert_into` : Δ ramené dans [0, 2π)).
3. **Accumulation robuste.** Les sources n'ont jamais à être choisies : chacune ajoute sa preuve à `L(Δ)`, pondérée en continu. Pour chaque source on mesure le résidu `r_i = |arg(p_i · e^{−jΔ̂})|` contre la dispersion attendue, et son poids est réduit en douceur quand le résidu grandit (pondération robuste type Cauchy/Huber, itérée 1 à 2 fois). Une source en désaccord (clic suspect) pèse moins, sans disparaître ; aucune source n'est éliminée ni sélectionnée.
4. **A priori.** Si aucune source n'est fiable à cet échantillon, `f̂(n)` est deviné par continuité à partir des échantillons voisins fiables, borné comme le noyau actuel ([1,5 MHz ; 0,75·`freq_hz`]).

Remarque : dans les zones propres, où toutes les sources s'accordent, l'estimateur du maximum de vraisemblance est mathématiquement une pondération par l'inverse de la variance ; c'est optimal et on ne peut pas faire mieux (gain ≤ 10·log₁₀(N) dB). L'intérêt de l'approche est dans les zones douteuses, où la preuve de chaque source est pondérée par sa fiabilité locale et par son accord avec les autres (étapes 3 et 4), au lieu d'une moyenne à poids fixes. Le seul cas binaire est `g_i = 0` hors couverture, en skip de disque ou en dropout franc.

## Insertion dans le code

`demod_block_cpu` (`crates/ld-decode/src/decode/demodblock.rs`) est découpé en trois :

- **Étage A, par source :** `fft → RFVideo → ifft` donne `z_i` (le signal analytique) et `Â_i`, σ_i.
- **Estimateur conjoint :** produit un unique `freq[]` (Hz) sur la grille de la primaire. Il remplace l'appel à `unwrap_hilbert_into`. Un seul `atan2` par échantillon, au lieu d'un par source ; `atan2` pèse ~80 % du coût de `unwrap_hilbert_into` d'après le code.
- **Étage B, une seule fois :** clip, FFT, filtres `FVideo`, de-emphasis, `demod_05`, `demod_burst` — inchangé.

Les appels à `demod_block_cpu` dans `decode/mod.rs` prennent déjà une tranche `&[f32]` ; c'est la jonction.

## Alignement

Le délai relatif τ_S(t_P) d'une source S par rapport à la primaire P est une fonction dynamique de la position. Trois composantes :

1. offset constant (départ des captures) ;
2. pente (horloges d'échantillonnage et vitesse moyenne de rotation différentes) ;
3. jitter de time-base fort : excentricité et oscillation du disque (retour à chaque tour), flutter, servo. Chaque lecteur a le sien, avec sa propre phase dans le tour ; aucun alignement entre captures n'est supposé. La plage de τ peut atteindre des dizaines à des centaines d'échantillons (à valider).

Estimation en coarse-to-fine, par source contre la primaire (topologie en étoile) :

- **Verrouillage grossier :** numéro de frame VBI (CAV/CLV) et `fileLoc` des champs.
- **Niveau 1, global :** pente et offset sur de longues fenêtres.
- **Niveau 2, ~1 ligne :** corrélation croisée par FFT sur des fenêtres de ~2000 échantillons avec recouvrement, pic interpolé au sous-échantillon, plage de recherche large.
- **Niveau 3, itératif :** on applique le warp, puis on re-corrèle sur des fenêtres plus courtes (~500).
- Lissage sous contrainte de régularité, état [τ, τ̇].

Hypothèse : aucun échantillon perdu dans une capture. La relation échantillon → temps est continue et monotone ; la seule cause de perte de verrouillage est un **skip de disque**.

### Application du décalage

Le décalage est une adresse de lecture dynamique : pour l'échantillon n de la primaire, on lit la source en `n + m_i(n)`.

- **Entier (défaut) :** `m = round(τ)`, zéro interpolation. Erreur de quantification ±0,5 échantillon (~7 ns RMS à 40 MHz), à peser sur les fronts rapides.
- **Fractionnaire (si le spike montre que l'entier pénalise le résultat) :** interpolation courte (~8 coefficients) de `p_i`/`f_i`, jamais du RF.

Critère d'acceptation : résiduel RMS d'alignement ≤ ~0,1 échantillon à 40 MHz (~2,5 ns).

### Skips de disque

Chaque source est découpée en segments de continuité (perte de corrélation, discontinuité du numéro de frame VBI). Au skip, la source passe à g_i = 0 puis est recalée sur la bonne position du disque (frame VBI, puis niveau 2). Sans VBI lisible : recherche par corrélation sur une fenêtre large, en respectant la parité de champ. Si c'est la primaire qui skippe, elle impose sa timeline comme dans un décodage normal.

## Rôle de la primaire

Elle fournit la grille temporelle, l'EFM et l'audio. Dans l'estimateur elle est traitée comme les autres sources (même pondération par fiabilité). Choix automatique par meilleur SNR mesuré sur un pré-passage court (quelques centaines de champs) ; `--stack-primary` pour forcer.

## Agnostique capture / lecteur

La fréquence instantanée est une grandeur physique du disque ; ni la fréquence d'échantillonnage, ni le format, ni le gain de la carte ne la modifient. Chaque source garde son `DecoderSpec` et son MTF. Les poids `g_i` sont fondés sur σ_i et `Â_i` mesurés sur la source elle-même, donc comparables d'une carte ou d'un lecteur à l'autre. Les niveaux Hz différents entre lecteurs sont corrigés par une calibration linéaire par segment avant l'estimation.

## Architecture d'exécution

Le décodeur démodule en parallèle et dans le désordre (prefetch), et `decode()` peut revenir en arrière (`redo`). Le délai doit donc être une fonction figée de la position :

1. **Passe d'alignement**, séquentielle et légère : table τ_S(t_P) par source (nœuds toutes les quelques centaines d'échantillons).
2. **Passe de démodulation conjointe**, parallèle : lit la table en lecture seule.

Coût de la passe 1 à trancher par mesure : (A) étage A + discriminateur complet, ou (B) discriminateur approximatif sur signal décimé.

Mémoire en O(N) ; l'étage A coûte N×, le discriminateur et l'étage B 1×.

## Contraintes

- **Parité préservée :** le chemin multi-source est opt-in (`--stack <fichier>`, répétable). Avec une seule entrée on garde le chemin actuel (`unwrap_hilbert_into`), donc des sorties byte-identiques aux références ; les tests de parité existants restent le garde-fou.
- **EFM** (dépend de la phase RF) et **audio** : pris sur la primaire dans un premier temps.
- **Mesure :** `VitsMetrics` existe déjà ; on compare une source seule, le stack TBC actuel, une moyenne de fréquences en sortie (baseline) et l'estimateur conjoint, y compris le nombre de clics FM résiduels.

## Plan

1. **Spike go/no-go :** paire synthétique à partir d'une capture (offset, pente ~20 ppm, wow, jitter à l'échelle de la ligne, bruit indépendant à fort niveau pour provoquer des clics, gain et fréquence d'échantillonnage différents, un skip). Mesurer (a) le résiduel d'alignement selon la fenêtre, la variante A/B et le mode entier/fractionnaire, (b) le nombre de clics résiduels et le SNR de l'estimateur conjoint contre la baseline.
2. Refactor étage A / discriminateur / étage B, avec parité N=1 vérifiée.
3. Lecture multi-source et verrouillage grossier.
4. Suiveur de délai, table τ, estimateur conjoint.
5. Option CLI et comparaison du SNR avec le stack TBC.
