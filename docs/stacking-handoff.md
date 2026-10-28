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

## 4 bis. Mesures sur de vraies captures (2026-10-27)

Six captures du même disque et du même lecteur (`D:\RF_CAPTURE\stack\stack_small_1..6.ddd.s16`, Domesday Duplicator, 75 MHz, s16, NTSC, 10 à 14 s chacune, mêmes réglages, même niveau de porteuse et même bruit hors bande). Les scripts de cette session sont dans le dossier temporaire de la session et ne sont **pas** dans le dépôt ; le code Rust n'est toujours pas modifié.

**Méthode qui marche, dans l'ordre :**
1. **Décalage grossier par le VBI.** Décoder chaque capture avec `ld-decode.exe --format s16 --inputfreq 75 <capture> <sortie>` (environ 30 s par capture) et lire le code de trame dans `vbi.vbiData[0]` des premiers fields de `.tbc.json` (BCD, 30 trames/s). Le décalage est `fileLoc_i − fileLoc_1` pour les codes communs (médiane ; dispersion ~0,2 ms). **Une corrélation seule ne suffit pas** : sur du contenu quasi répétitif, ma corrélation sur 10 s s'est trompée de 14, 28 et 14 trames entières pour trois captures sur cinq.
2. **Raffinage local à la ligne près** : corrélation du signal démodulé décimé à 1 MHz, fenêtre de 60 ms, recherche ±600 µs autour du décalage VBI.
3. **Courbe de vitesse** : décalages sur fenêtres de 16384 échantillons (R = 6000), repliés modulo la ligne (4767 échantillons à 75 MHz) puis lissés. Le délai relatif varie de plusieurs centaines d'échantillons en 40 ms ; le suivi doit rester près de la courbe, sinon il saute sur la ligne voisine.
4. **TBC par fronts de synchro** : front descendant de chaque ligne (seuil 7,8 MHz sur le signal passe-bas), délai de chaque capture par corrélation sur ±512 échantillons, interpolé entre les fronts. Le délai sert d'**adresse de lecture** `n + τ(n)`, sans ré-échantillonner le RF. Ensuite la phase de la sous-porteuse couleur peut affiner (gain de 1 à 2 IRE sur un résiduel mal calé ; sans objet quand l'alignement est déjà bon).

**Résultats (fenêtre de 80 ms, signal démodulé passe-bas 4,2 MHz, moyenne simple, IRE NTSC) :**
- Les 15 paires parmi les 6 captures sont à **2,3 IRE RMS**, soit **1,6 à 1,7 IRE de bruit par capture** (méthode des trois captures étendue à six, résidu d'ajustement 0,11 IRE²).
- Bruit de l'empilement mesuré contre une capture mise de côté (bruit de celle-ci soustrait) : 1 source 1,65 ; 2 sources 1,17 (−3,0 dB) ; 3 sources 0,95 (−4,8 dB) ; 4 sources 0,82 (−6,0 dB) ; 5 sources 0,74 (−7,0 dB). Le gain théorique en 10·log10(N) est atteint.
- Le même alignement sur un seul tirage de 80 ms à deux positions (t = 3 s et 4 s) donne des valeurs stables.

**Pistes écartées en cours de route :** décalages par lignes ou trames entières appliqués à l'aveugle, fenêtres courtes de corrélation (< 1 ligne, pires), correction par la phase de couleur seule, ré-échantillonnage ligne par ligne (même résultat que l'adresse de lecture).

## 4 ter. Exploration des méthodes de démod multi-source (2026-10-27)

Mêmes six captures, fenêtre de 3 M d'échantillons (40 ms) vers t = 3 à 4 s. Prototype numpy uniquement ; le décodeur Rust n'est pas modifié. Les scripts sont restés dans le dossier temporaire de la session (`bcommon.py` en est le module commun) et ne sont **pas** dans le dépôt.

### Méthode de mesure

Toutes les méthodes sont jugées **sur la sortie démodulée** (baseband), au même filtre de sortie (passe-bas 4,2 MHz), avec trois indicateurs complémentaires :

- **Erreur contre une référence tenue à l'écart** : moyenne des captures non utilisées par la méthode, bruit de la référence soustrait. Ce bruit doit être estimé **sur les lignes d'image**, pas sur les lignes noires : l'erreur indépendante d'une capture y vaut 1,65 IRE contre 1,22 sur les lignes noires (bruit FM dépendant du niveau, gigue visible seulement sur le détail). L'estimer sur les lignes noires avait fait croire à un plancher à N = 4 qui n'existe pas.
- **Réponse du signal |H| par bande** (spectre croisé avec la référence) : |H| < 1 signale un lissage. Une méthode qui baisse le RMS en baissant |H| ne fait que filtrer.
- **Comparaison à réponse égale** : la méthode contre la démod normale suivie du filtre linéaire qui reproduit la même réponse. Seul l'écart qui subsiste est un vrai gain de démodulation.
- **Bruit sur les lignes noires**, séparé en part commune et part indépendante par l'ajustement `var(moyenne de n captures) = c + i/n`. La référence tenue à l'écart ne voit pas les erreurs communes, ce qui rend cette mesure indispensable.

Pour tester sans référence, deux empilements disjoints de même taille N se comparent directement : `var(A − B)/2` est l'erreur indépendante d'un empilement de N.

Les tests à bas CNR ajoutent un bruit gaussien indépendant à chaque capture (« CNR ajouté », convention de `sweep.py`). Les défauts propres à un passage sont simulés par des dropouts de 2 à 27 µs (RF × 0,03), indépendants par capture.

### Ce qui est retenu

| Méthode | Gain mesuré | Domaine d'emploi |
|---|---|---|
| **Moyenne pondérée** par 1/σ² | Erreur indépendante exactement en 1/√N (empilements disjoints : 1,651 / 1,167 / 0,953 IRE pour N = 1, 2, 3), |H| ≈ 1 | Base de l'empilement. Pour du bruit gaussien indépendant, c'est l'estimateur optimal |
| **Combinaison cohérente pondérée par l'amplitude** (phaseurs recalés sur la porteuse, poids ∝ amplitude/σ²) | Avec 1 à 3 % de dropouts par passage, N = 2 : 4,36 → 1,78 IRE (1 %), 6,93 → 2,13 IRE (3 %) ; dans les défauts 28 → 8 IRE. Identique à la moyenne sur captures propres | Défauts propres à un passage, surtout à N = 2 où il n'y a pas de majorité |
| **Médiane ou poids de cohérence InSAR** | Avec défauts, N = 4 : 0,96 à 1,00 IRE (niveau d'un empilement sans défaut), contre 2,7 à 5,1 pour la moyenne | À partir de 3 sources |
| **Régression de la diaphonie audio après démod** (un coefficient complexe par porteuse, 2,30114 et 2,8125 MHz, régresseurs tirés du RF de la capture) | Bruit des lignes noires d'une capture 2,46 → 2,07 IRE (−1,6 dB), part commune 2,11 → 1,64 IRE, |H| = 1 | Toujours, par capture. **Uniquement après démod** (voir écarté) |
| **Correction AM→PM** apprise sur les différences entre captures (enveloppe relative à son niveau attendu pour la fréquence, petit filtre, terme de niveau) | Version linéaire : bruit indépendant sur l'image 1,695 → 1,481 IRE (−1,2 dB), |H| ≈ 1. Version **non linéaire** : −1,50 dB sur des captures **non vues** à l'apprentissage, neutre sur l'image | Appliquée à l'**écart** de chaque capture à l'enveloppe moyenne. Appliquée à toute l'enveloppe, elle fait monter la part commune (1,64 → 2,05 IRE) |
| **Filtre Wiener RF avant discriminateur** (poids SNR du RF, force 0,5 à 1) | −0,30 à −0,34 dB à réponse égale | **Une capture seule uniquement** : sur un empilement RF il nuit (+0,2 à +0,7 dB), appliqué par source avant un empilement baseband il ne rapporte plus rien |

Démodulation directement sur une base de temps corrigée (lecture des sources aux positions d'une grille idéale de 910 points par ligne) : fonctionne et évite le rééchantillonnage après démod. Gigue équivalente par capture 2,45 ns (fronts de synchro) et 2,32 ns (fronts + salve), contre 2,47 ns pour le `.tbc` de ld-decode ; l'empilement de 3 sources ramène l'erreur de timing à 1,42 ns (≈ 2,45/√3). Estimer une base de temps commune à toutes les sources n'apporte presque rien (2,39 ns) : l'erreur restante est une gigue à l'intérieur des lignes, propre à chaque passage.

Alignement à bas CNR : aligner sur la sortie démodulée de chaque source plutôt que sur le discriminateur bruité, puis choisir le lissage des délais ligne à ligne **par cohérence** (le lissage qui minimise l'écart de la source alignée avec la référence). Ce critère choisit σ = 1 ligne sur les captures propres (optimum, 2,15 IRE) et 4 à 8 lignes à bas CNR, sans vérité terrain. Un lissage fixe fort (σ = 4) efface du vrai jitter : sur captures propres, l'écart entre paires passe de 2,29 à 4,33 IRE.

### Ce qui est écarté, et pourquoi

| Piste | Résultat |
|---|---|
| Filtre de Kalman (avec lissage arrière) | Gain RMS apparent, mais |H| tombe à 0,48–0,90 sur 3–4,2 MHz : il retire de la chroma. À réponse égale, identique au discriminateur jusque dans les queues de distribution. La baisse des clics était un effet du lissage |
| Estimateur conjoint (phaseurs) | Même RMS que la moyenne, mais |H| = 0,93–0,98 dans toutes les bandes à bas CNR : perte de contraste de 3 à 7 % |
| Combinaison cohérente sur captures propres | Identique à la moyenne pondérée (≤ 0,3 %) |
| Niveaux connus de synchro et de suppression imposés | Gain uniquement sur les segments imposés ; 0,2 % sur l'image |
| Suréchantillonnage (discriminateur à décalage de 2 ou 3 échantillons) | Identique à un filtre linéaire ; au-delà, la phase se replie |
| Prédiction 3D comme arbitre (lignes ±2) | Version douce : lissage déguisé (|H| 0,79–0,94), égale à un filtre. Version franche : écarte 0 à 0,05 % des échantillons, aucun effet |
| Remodulation itérative de l'empilement comme référence | Améliore toujours un peu, au plus −0,1 dB à 9 dB, converge en une itération. En phase, chaque capture n'est que la référence plus du bruit blanc. Les poids par échantillon tirés de cette référence dégradent légèrement |
| Poids variables par ligne (« lucky imaging ») | Validés avec une référence tenue à l'écart : 1,000 IRE contre 0,956 pour des poids fixes |
| Correction de MTF dans le RF, déconvolution multicanal | Identique à un post-filtre de même réponse ; les six passages ont la même netteté à ±1 % |
| Retirer les porteuses audio dans le RF (coupe-bande du décodeur ou soustraction d'un modèle remodulé) | Aucun effet sur la diaphonie : l'audio module la **phase** de la porteuse vidéo, la correction doit se faire après démod. Modèles de couplage plus riches (niveau, filtre, produits, couplage variable) : pas mieux qu'un coefficient |
| Harmonique 2 de la porteuse comme seconde observation | 36 dB sous la fondamentale, au plancher de bruit |
| Diaphonie EFM (régression sur la bande < 1,9 MHz) | Pas de fuite linéaire ; la régression ajoute 2 à 4 dB de bruit |
| Porteuses audio comme capteurs de gigue (leur phase devrait suivre la base de temps) | Réfuté : elles ne donnent pas une mesure de gigue exploitable pour l'alignement |
| Poids selon la longueur de ligne de chaque capture (avant TBC, une capture plus lente a plus d'échantillons par détail) | Les longueurs de ligne ne diffèrent que de 0,01 % entre captures : 0 dB |
| Modèle d'enveloppe commune (le bruit commun vu comme une enveloppe partagée, à retirer) | Fonctionne au niveau de suppression seulement ; sur l'image il dégrade, faute de zones plates pour apprendre le modèle |
| Masque temps-fréquence piloté par le guide (filtre suiveur : démélange par la phase du guide, passe-bas de 0,5 à 4 MHz, reconversion, puis discriminateur ; équivalent d'un masque STFT sur la crête) | Testé de 13 à 3 dB et sur captures propres, délais idéaux. Guide = l'empilement : 0,00 dB à réponse égale, même à 3 dB où il retire plus de 5 000 clics. Guide = Kalman : baisse d'erreur obtenue en perdant de la chroma (|H| 0,49 à 0,90). Guide passe-bas 2,5 MHz : chroma effacée. Hors clics, suivre la crête est un filtre linéaire |
| Conversion FM → AM par FFT (rampe spectrale, dérivateur) | Même estimateur que le discriminateur (`Im(z'·z*)/|z|²`) ; pas testé, équivalence mathématique |

### Architecture guide / sortie

Règle : **aucune réduction de bruit dans la sortie**, mais elle est permise dans une estimation qui ne sert qu'à décider. D'où deux couches :

| Couche | Contenu | NR permise |
|---|---|---|
| **Guide** (estimation) | Empilement cohérent dans le domaine de l'amplitude (qui lisse bien le bruit), Kalman (q = 1e-3), passe-bas 2,5 MHz, prédiction | Oui |
| **Décisions tirées du guide** | Délais et base de temps, poids de fiabilité, défauts, apprentissage des corrections (audio, AM→PM) | – |
| **Sortie** | Combinaison cohérente pondérée par l'amplitude des **vrais échantillons**, lus aux délais décidés, plus les corrections déterministes par capture | Non |

On garde ainsi l'avantage de l'empilement AM (lissage du bruit) pour décider, et celui de l'empilement FM (moins d'artefacts de démod) dans le résultat.

Premier usage testé, l'alignement à bas CNR, qui était la plus grosse perte : des délais faux de 2 à 3 échantillons brouillent la chroma dans la moyenne (|H| = 0,82 à 0,94 sur 3–4,2 MHz).

1. **Aligner chaque source sur le guide débruité** au lieu de la sortie de la source : erreur de délai 2,36 → 1,55 échantillon à 11 dB, mais le RMS bouge peu.
2. **Affiner les délais par la phase de la sous-porteuse couleur**, mesurée entre chaque source et l'empilement : signal analytique de la bande 2,9–4,3 MHz, corrélation croisée sur une fenêtre de 2048 échantillons, phase convertie en délai (× FS / 2π·fsc), lissée sur 1 à 4 lignes avec la quantité de couleur comme poids, 2 itérations.

Erreur en IRE contre la référence tenue à l'écart, N = 4, |H| de la chroma entre parenthèses :

| CNR ajouté | Délais idéaux | Fronts de synchro | **+ phase couleur** |
|---|---|---|---|
| 13 dB | 10,64 (1,00) | 11,87 (0,965) | **11,09 (0,996)** |
| 11 dB | 13,37 (1,00) | 15,33 (0,923) | **14,18 (0,988)** |
| 9 dB | 16,87 (1,00) | 19,92 (0,865) | **18,16 (0,970)** |
| Captures propres | 0,96 | 1,17 | 1,17 (neutre) |

Gain de 0,6 à 1 dB à bas CNR, chroma presque entièrement retrouvée, aucun coût sur captures propres. La sortie reste une combinaison des vrais échantillons.

Pas encore testé avec le guide : apprentissage de la diaphonie audio et de l'AM→PM sur le guide plutôt que sur la moyenne brute. Les poids de fiabilité par bande sont testés ci-dessous.

Définition retenue de la NR exclue : **perdre du signal vrai pour gagner en bruit**. Un traitement de sortie qui retire du bruit en gardant |H| ≈ 1 dans toutes les bandes, avec un gain qui tient à réponse égale, est permis.

### Poids par sous-bande RF (idée des ondelettes)

But : garder un avantage propre au côté FM, sinon la même chose peut se faire après démod dans un stacker de `.tbc` (ld-disc-stacker). Chaque capture alignée et calée en phase est découpée, **avant le discriminateur**, par un banc de filtres à partition de l'unité (bandes gaussiennes de 1 MHz centrées de 3 à 13 MHz, sur la bande analytique 2,5–13,5 MHz). Dans chaque bande et à chaque instant (fenêtre T = 256 échantillons, 3,4 µs), chaque capture reçoit un poids ; les bandes sont additionnées puis démodulées une seule fois.

Poids testés :

- **Amplitude relative** : amplitude de la bande comparée à la médiane des sources, pénalisée dans les deux sens (`min(r, 1/r)^p`, p = 2), pour la perte de signal comme pour l'interférence.
- **Bandes prédites par le guide** (jeu entre FM et AM) : la fréquence instantanée du guide prédit l'amplitude que chaque bande devrait avoir (table apprise sur la fenêtre, médiane par pas de 0,1 MHz de fréquence) ; une capture qui s'en écarte perd du poids dans cette bande. Ce poids juge une source **même à N = 2**, où une médiane n'est qu'une moyenne.
- **Hybride** : poids prédit × accord avec la médiane complexe des **autres** sources dans la même bande (`min(1, D_med/D)^pc`, D l'écart local à cette médiane). À N = 2 il se réduit aux bandes prédites.
- Comparés à la combinaison cohérente retenue, à la médiane après démod (ce que fait un stacker) et à la même fusion par sous-bandes faite après démod.

Défauts simulés, propres à chaque passage, sur 3 % du temps (segments de 2 à 27 µs) : dropout (RF × 0,03), haut de bande RF atténué de 20 dB au-dessus de 9,5 MHz, interférence (raie à 5–12 MHz, amplitude 0,7 × la porteuse). Référence : captures propres 5 et 6. Erreur en IRE **dans les défauts** (erreur totale pour les captures propres) :

| N = 2 | Après démod (moyenne) | RF retenu | **RF bandes prédites** |
|---|---|---|---|
| Captures propres | 1,279 | 1,281 | 1,283 |
| Dropout | 27,46 | 4,41 | **3,38** |
| Haut de bande atténué | 1,77 | 1,76 | **1,58** |
| Interférence | 46,62 | 47,73 | **9,41** |

| N = 4 | RF retenu | Médiane après démod | Sous-bandes après démod | RF bandes prédites | RF amplitude relative | **RF hybride pc = 1** |
|---|---|---|---|---|---|---|
| Captures propres | 0,959 | 0,962 | 0,992 | 0,961 | 0,955 | **0,958** |
| Dropout | 1,58 | **1,06** | 1,39 | 2,03 | 1,45 | 1,40 |
| Haut de bande atténué | 1,17 | 1,12 | 1,08 | 1,16 | 1,06 | **1,06** |
| Interférence | 23,99 | 3,30 | **1,85** | 5,47 | 4,77 | 3,13 |

Lecture :

- **À N = 2, le côté FM gagne partout**, et après démod on ne peut rien faire (une médiane de 2 est une moyenne). C'est l'avantage propre au RF : interférence divisée par 5, dropouts −2,3 dB par rapport à la combinaison retenue. Réserve : sur les dropouts, |H| tombe à 0,975 sur 1–2 MHz (petite perte de signal, à surveiller).
- **À N = 4, l'hybride fait jeu égal avec un stacker** : meilleur sur la perte du haut de bande (invisible après démod), un peu meilleur que la médiane sur l'interférence mais pas que les sous-bandes après démod, moins bon sur les dropouts (1,40 contre 1,06). Les deux facteurs se complètent (accord seul : 3,57 / 1,08 / 3,15). pc = 2 est moins bon, pc = 4 se dégrade (dropouts 3,72).
- Coût nul sur captures propres, |H| ≈ 1 partout.
- Faiblesse probable sur les dropouts à N = 4 : la décision sur 256 échantillons réagit trop tard aux bords francs, alors que la médiane après démod tranche échantillon par échantillon. Non vérifié.
- La cohérence InSAR par bande (sans prédiction) échoue sur l'interférence (20,35 dans les défauts à N = 4) : dans une bande où le signal est absent, toutes les sources ont une cohérence faible et la source brouillée n'est pas distinguée.

Recette qui se dessine : à N = 2, bandes RF prédites par le guide ; à N ≥ 3, hybride avec pc = 1. Les défauts sont **simulés** ; tous les dropouts réels de ces captures sont communs aux six.

### Ce qui reste

- **Le bruit commun aux captures domine.** Sur les lignes noires du `.tbc` de ld-decode : 0,95 IRE par capture, dont 0,72 commun et 0,62 indépendant (57 % commun) ; six captures ne gagnent que −1,9 dB sur le bruit total. Le commun se concentre sous 0,5 MHz et sur les raies audio.
- Ce bruit commun est **indépendant d'une trame à l'autre** (écart entre trames k et k+2 d'une même capture : 0,957 IRE, soit tout le bruit) : c'est du bruit des pits. Aucun empilement de captures du même disque ne le retire ; seule une moyenne entre trames le ferait, ce qui reviendrait à du 3DNR, exclu de la sortie.
- Tous les dropouts réels de ces captures sont présents dans les six captures (défauts du disque). Les gains sur défauts propres à un passage viennent de dropouts **simulés** et sont à confirmer sur un disque abîmé.

## 5. Ce qui n'est PAS établi

- Les gains de la section 4 bis sont calculés avec un bruit estimé sur les **mêmes données**. La section 4 ter les confirme sans référence (empilements disjoints en 1/√N) et compare l'estimateur conjoint, la combinaison cohérente et d'autres méthodes.
- Des fenêtres de 40 à 80 ms sur 10 à 14 s de captures, un seul tirage de bruit ajouté par cas ; pas de test du décodeur Rust, donc ni sync, ni couleur décodée, ni coût. Les dropouts testés sont simulés.
- La régression audio et la correction AM→PM sont mesurées dans le prototype, sans la désaccentuation ni les filtres du décodeur ; leur gain dans le `.tbc` final reste à confirmer.
- Hypothèse NTSC pour l'échelle IRE (7,8 MHz comme seuil de synchro, 4767 échantillons/ligne) ; non revérifiée sur du PAL.
- Avant les vraies captures, les tests utilisaient un seul fichier réel avec des copies déformées par mes modèles de time-base, dont le bruit d'origine est commun (chapitre 4). Le jitter réel entre captures du même lecteur est maintenant mesuré (délai relatif de quelques centaines d'échantillons en 40 ms, erreur ligne à ligne de quelques échantillons) mais pas caractérisé sur une plus longue durée.
- Un **seul tirage** par cas synthétique, bruit gaussien blanc, modèles de jitter inventés (lent : corrélé sur ~50 µs ; rapide : ~8 µs).
- La forme des poids `g_i` est une approximation non optimisée.
- Le coût réel en Rust (passe d'alignement, N× l'étage A) n'est pas mesuré.
- Le ré-encodage d'un `.tbc` en FM avec time-base déformée n'est pas fait (il demande une pre-emphasis cohérente avec la de-emphasis du décodeur).
- La combinaison cohérente avant discriminateur est maintenant testée (section 4 ter) : sans intérêt sur captures propres, mais la meilleure méthode contre les défauts propres à un passage avec 2 sources.
- L'affinage par la phase couleur n'est mesuré qu'avec du bruit **ajouté** sur une fenêtre de 40 ms ; non vérifié sur des captures réellement faibles, ni sur du PAL (autre fsc), ni sur des scènes sans couleur, où il doit retomber sur l'alignement par fronts de synchro.
- Les poids par sous-bande RF ne sont jugés que sur des défauts **simulés**, avec un seul tirage par cas et des paramètres (T, p, pc) choisis sur ces mêmes données.

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

1. ~~Mesurer le résiduel d'alignement sur de vraies paires~~ : fait (section 4 bis). Reste à le faire sur des fenêtres plus nombreuses et plus longues, y compris autour des dropouts et des sauts de trame.
2. **Alignement fin** : les fronts de sync donnent déjà un alignement tel que les 15 paires sont à 2,3 IRE. Reste à mesurer le résiduel en échantillons (cible ≤ 0,25) et à vérifier que l'alignement tient en cas de jitter rapide.
3. **Garde de confiance** : pondérer chaque source par la qualité de son alignement local (par exemple la corrélation entre la primaire et la source recalée) pour qu'un alignement douteux ramène son poids vers 0. Indispensable : sans elle, un décalage grossier faux (comme les 14 et 28 trames d'erreur de la corrélation seule) ou un jitter rapide dégrade l'image.
4. **Mode fractionnaire** d'office. **Décalage grossier par le VBI** (section 4 bis) et non par corrélation seule ; garder le calcul des trames communes dans la passe d'alignement.
5. ~~Tester la combinaison cohérente avant discriminateur~~ : fait (section 4 ter).
6. **Confirmer sur un disque abîmé** les gains de la combinaison pondérée par l'amplitude et de la médiane (dropouts réels propres à un passage).
7. Refactor Rust (étage A / estimateur / étage B avec parité N=1), vérification VBI, table de délais, option CLI, audio puis EFM. Ce qu'il faut y mettre, d'après la section 4 ter :
   - structure guide / sortie : un guide (empilement cohérent, éventuellement débruité) qui ne sert qu'aux décisions, une sortie sans NR ;
   - alignement : VBI, courbe de vitesse, fronts de synchro, lissage des délais choisi par cohérence, puis affinage par la phase couleur contre l'empilement, lecture à `n + τ(n)` ;
   - combinaison : cohérente pondérée par l'amplitude, qui retombe sur la moyenne pondérée quand tout va bien ; contre les défauts propres à un passage, poids par sous-bande RF (bandes prédites par le guide à N = 2, hybride pc = 1 à partir de 3 sources) ;
   - par capture : régression de la diaphonie audio après démod, correction AM→PM non linéaire sur l'écart à l'enveloppe moyenne ;
   - à ne pas implémenter dans la sortie : Kalman, estimateur conjoint, poids par ligne, prédiction 3D, correction de MTF RF (le Kalman reste possible dans le guide).
8. Vérifier dans la bande RF exacte du décodeur si ses coupe-bande audio servent à quelque chose : contre la diaphonie dans la vidéo, ils sont sans effet.
9. Étendre l'usage du guide : apprentissage des corrections sur le guide (section 4 ter, architecture guide / sortie).
10. Poids par sous-bande RF : décision plus rapide aux bords des dropouts (amplitude large bande à court terme) ; puis confirmation sur un disque abîmé.

**Partage avec ld-disc-stacker (décision du 2026-10-28).** Le décodeur multi-source ne garde que ce qui a besoin du RF ou qui renvoie une information au décodage : alignement et lecture aux positions corrigées, combinaison cohérente et poids par sous-bande avant le discriminateur, guide et ses décisions, corrections par capture tirées du RF (diaphonie audio, AM→PM). Tout traitement fait sur la vidéo démodulée sans retour vers le décodage (médiane ou moyenne après démod, fusion par sous-bandes de la vidéo, choix de dropouts sur le `.tbc`) est laissé à ld-disc-stacker. En particulier, l'avantage de la médiane après démod sur les dropouts à N ≥ 3 n'est pas à reproduire dans le décodeur.

## 8. Fichiers

- `docs/stacking-design.md` : conception complète et résultats détaillés.
- `docs/stacking-handoff.md` : ce document.
- `scripts/stacking_spike/stack_spike.py` : générateur synthétique, suiveur de délai, estimateurs, scoring.
- `scripts/stacking_spike/sweep.py` : balayage synthétique avec erreur d'alignement injectée.
- `scripts/stacking_spike/real_spike.py` : même test sur un vrai fichier s16.
- Les scripts des sections 4 bis et 4 ter (`bcommon.py`, `bbtest.py`, `tbcdemod.py` et les bancs d'essai, dont `guide_align.py` et `guide_chroma.py` pour l'architecture guide / sortie, `tfmask.py` pour le masque temps-fréquence, `subband.py`, `subband2.py` et `hybrid.py` pour les poids par sous-bande) ne sont pas versionnés : ils sont restés dans le dossier temporaire de la session. Ils supposent les captures de `D:\RF_CAPTURE\stack` et un environnement conda avec `numpy` et `numba`.
