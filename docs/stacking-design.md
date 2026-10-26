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

### Vérification VBI, source par source et frame par frame

Avant tout suivi fin, chaque source est vérifiée **séparément** contre la primaire, frame par frame :

1. **Table de la source :** pour chaque frame, numéro lu dans le VBI (CAV : numéro de frame ; CLV : timecode), position `fileLoc` de ses deux champs, parité de champ (`is_first_field`). Elle est construite par un balayage de la source seule avec `Decoder::scan_field`, qui renvoie déjà `frame_number` ; `FieldInfoEntry` porte aussi `file_loc`, `vbi` et `is_first_field`.
2. **Contrôle :** pour chaque frame de la primaire, on retrouve la même frame dans la source et on vérifie que le numéro correspond, que les deux champs sont dans le bon ordre, et que la position est cohérente avec la frame précédente.
3. **Résultat par frame :** `ok` (la source contribue), ou `non couverte` (frame absente, répétée ou sautée : skip de disque, arrêt sur image, pause), ou `douteuse` (VBI illisible).
4. Seules les frames `ok` entrent dans l'estimateur ; les autres ont `g_i = 0`.

Une lecture VBI ratée ne prouve pas un désaccord. Le numéro doit avancer de 1 par frame : un numéro isolé incohérent entre deux voisins cohérents est traité comme une erreur de lecture (la frame reste `ok` par interpolation), et un vrai saut de séquence est un skip. Un disque sans numéros lisibles (lead-in, lead-out, zones non codées) retombe sur la corrélation croisée sur fenêtre large, avec respect de la parité de champ.

Cette vérification est faite une fois en amont (passe d'alignement), pas pendant la démodulation parallèle.

### Skips de disque

Chaque source est découpée en segments de continuité (perte de corrélation, discontinuité du numéro de frame VBI). Au skip, la source passe à g_i = 0 puis est recalée sur la bonne position du disque (frame VBI, puis niveau 2). Sans VBI lisible : recherche par corrélation sur une fenêtre large, en respectant la parité de champ. Si c'est la primaire qui skippe, elle impose sa timeline comme dans un décodage normal.

## Audio analogique et EFM

Ces deux sorties sortent du même noyau (`demod_block_cpu`) à partir du **même** spectre `indata` que la vidéo, donc l'étage A par source les fournit presque gratuitement.

### Audio analogique (même estimateur, phase 1)

Dans le noyau, chaque canal (gauche, droite) est une démodulation FM : tranche du spectre autour de la porteuse audio, multipliée par `filt1`, `ifft` vers un signal analytique décimé `a1` (cadence `a1_freq`, décimation `audio_fdiv`), puis `unwrap_hilbert(a1) + low_freq`. C'est exactement la structure vidéo, à cadence plus basse. L'estimateur conjoint s'applique tel quel : `p_i = a1_i(k)·conj(a1_i(k−1))`, accumulation pondérée par la fiabilité de chaque source, un seul `atan2`.

- **Alignement :** le signal audio ne dépasse pas ~20 kHz, donc il tolère très bien l'erreur de délai (un décalage de l'ordre de l'échantillon décimé est négligeable). Le décalage entier dynamique `m_i(n)`, ramené à la cadence décimée, suffit.
- **Coût :** deux petites `ifft` par source et par canal, déjà présentes ; un `atan2` par échantillon décimé.
- **En aval :** l'étage 2 (`audio.rs` : filtre, de-emphasis, suppression des pics par `findpeaks`, rééchantillonnage vers le `.pcm`) reste inchangé et reçoit un étage 1 plus propre.

### EFM (phase 2, plus délicat)

L'EFM est `ifft(indata · Fefm).real`, un signal **linéaire** (une forme d'onde réelle, pas de FM), écrêté en i16, puis `efm_pll.rs` en détecte les passages à zéro (interpolation sous-échantillon) pour produire les T-values. Seule la **position temporelle** des passages à zéro porte l'information.

- **Accumulation :** somme pondérée des formes d'onde alignées, `ê = Σ g_i·e_i / Σ g_i`, faite **avant** l'écrêtage i16 et la détection des passages à zéro. C'est l'estimateur optimal pour un bruit gaussien indépendant.
- **Alignement exigeant :** une erreur de délai se convertit directement en gigue des passages à zéro, donc le mode fractionnaire est probablement nécessaire ici (à valider).
- **Calibration par source :** gain, décalage continu, et surtout réponse différente d'un lecteur à l'autre. Il faut sans doute une égalisation adaptative (petit filtre FIR ajusté par LMS contre la primaire) avant la somme.
- **Polarité et asymétrie :** à vérifier par source ; un décalage de niveau déplace les passages à zéro.

C'est pourquoi l'EFM vient après la vidéo et l'audio.

## Rôle de la primaire

Elle fournit la grille temporelle. Pour l'audio et l'EFM elle est aussi traitée comme les autres sources dès que leurs phases sont implémentées. Dans l'estimateur elle est traitée comme les autres sources (même pondération par fiabilité). Choix automatique par meilleur SNR mesuré sur un pré-passage court (quelques centaines de champs) ; `--stack-primary` pour forcer.

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
- **Audio et EFM :** pris sur la primaire tant que leur phase n'est pas implémentée (voir [Audio analogique et EFM](#audio-analogique-et-efm)). Avec une seule entrée, leurs chemins actuels restent inchangés.
- **Mesure :** `VitsMetrics` existe déjà ; on compare une source seule, le stack TBC actuel, une moyenne de fréquences en sortie (baseline) et l'estimateur conjoint, y compris le nombre de clics FM résiduels.

## Résultats du spike (prototype numpy, signal synthétique)

Code : `scripts/stacking_spike/` (`sweep.py <cnr,cnr,...> <corrélation_jitter> [seed]`). Signal FM NTSC synthétique à 40 MHz, une capture par source avec son propre offset, sa pente (±20 ppm × i), son wobble à ~30 Hz (3 et 4 µs), son jitter (30 ns RMS) et son bruit blanc gaussien. Erreur = RMS en IRE après un passe-bas de 4,2 MHz, contre la fréquence vraie. **Un seul tirage, bruit gaussien idéal, images aléatoires : ce n'est pas du vrai LaserDisc, et le code du dépôt n'a pas été exécuté.** Le prototype n'est pas le code Rust.

**Gain avec alignement parfait (délai vrai connu).** 25 dB de CNR : 6,77 IRE pour une source, 4,76 pour deux (2,9 dB), 3,90 pour trois (4,8 dB). C'est le gain théorique 10·log₁₀(N).

**Sensibilité à l'erreur d'alignement (erreur lisse injectée sur le vrai délai, 2 sources) :**

| CNR | erreur 0 | 0,25 smp | 0,5 smp | 1,0 smp | source seule |
| --- | --- | --- | --- | --- | --- |
| 25 dB | 4,76 | 4,80 | 4,91 | 5,31 | 6,77 |
| 35 dB | 1,57 | 1,68 | 1,98 | 2,83 | 2,18 |

À 35 dB, une erreur de 0,5 échantillon fait perdre ~70 % du gain, et 1 échantillon est **pire que sans stacking**. Le critère de 0,1 échantillon est justifié ; viser au plus 0,25.

**Précision du suiveur de délai testé** (corrélation croisée normalisée sur le signal démodulé passe-bas, coarse-to-fine, niveaux 16384 puis 2048 échantillons, lissage) : résiduel RMS de 0,25 échantillon à 35 dB, 0,52 à 25 dB, ~2,5-3 à 12-14 dB (le bruit dégrade la corrélation). Avec un jitter plus rapide que la ligne (corrélé sur ~8 µs), 1,4 échantillon à 25 dB. À 35 dB le suiveur conserve ~76 % du gain (1,69 contre 1,57 optimal). **Il n'atteint donc pas le critère de 0,1 et reste le principal point faible.**

- Des fenêtres de 1024 échantillons ou moins ne contiennent pas assez d'image pour corréler.
- Un affinage par retard de groupe sur le spectre croisé des signaux analytiques a été essayé et **ne fonctionne pas** : même sur un test propre à décalage connu l'erreur reste de 0,7 à 1,3 échantillon, et il est inobservable dans les zones où la porteuse est quasi pure. Retiré du prototype. Pistes non testées : alignement par les fronts de sync de chaque source, ou corrélation sur un signal moins filtré.

**Estimateur conjoint contre moyenne pondérée par 1/σ² (baseline) : aucun avantage mesuré en RMS.** Pour les trois CNR testés (25, 14 et 12 dB) les deux sont à moins de 1 % l'un de l'autre, la moyenne pondérée étant très légèrement meilleure (17,0 contre 16,9 à 14 dB). Le seul écart visible est le nombre de clics FM résiduels, dans un premier run à 12 dB et 2 sources : 44-59 pour la moyenne contre 10-16 pour l'estimateur conjoint (source seule : 1994) ; à 3 sources il n'y en a plus aucun. La pondération robuste n'apporte rien de mesurable, et la médiane (N = 3) est moins bonne que la moyenne. Ces résultats viennent d'un unique tirage et des approximations de poids décrites plus haut, donc ils ne tranchent pas.

**Autres constats :** le mode entier coûte 0,09 IRE à 25 dB (4,85 contre 4,76) et 0,14 à 35 dB (1,71 contre 1,57), ce qui plaide pour le fractionnaire. La correction de cadence locale `exp(j·Δ0·τ')` (différence de vitesse de lecture entre sources) apporte peu dans ce modèle (0,02 IRE).

**Conséquence :** l'alignement est le facteur limitant, pas la forme de l'estimateur. Reste à tester, hors périmètre convenu pour l'instant : une combinaison cohérente avant discriminateur, qui améliore la porteuse elle-même avant la non-linéarité et pourrait battre la moyenne près du seuil FM.

## Plan

1. **Spike go/no-go :** paire synthétique à partir d'une capture (offset, pente ~20 ppm, wow, jitter à l'échelle de la ligne, bruit indépendant à fort niveau pour provoquer des clics, gain et fréquence d'échantillonnage différents, un skip). Mesurer (a) le résiduel d'alignement selon la fenêtre, la variante A/B et le mode entier/fractionnaire, (b) le nombre de clics résiduels et le SNR de l'estimateur conjoint contre la baseline.
2. Refactor étage A / discriminateur / étage B, avec parité N=1 vérifiée.
3. Lecture multi-source et verrouillage grossier.
4. Suiveur de délai, table τ, estimateur conjoint vidéo.
5. Option CLI et comparaison du SNR avec le stack TBC.
6. Audio analogique : même estimateur sur `a1`, validation sur le `.pcm`.
7. EFM : accumulation de formes d'onde avec calibration/égalisation par source, validation par le taux d'erreurs de T-values.
