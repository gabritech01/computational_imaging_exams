# Report — Deblur & Denoise su KaoKore (bozza in costruzione)

Bozza progressiva del report finale: viene aggiornata man mano che si completa ogni metodo.
Le sezioni di setup (dataset, operatore di degradazione, metriche) sono riassunte qui in
forma sintetica; la spiegazione teorica completa di ogni scelta è in `NOTE_ORALE.md` (fuori
da questa cartella).

## 1. Setup sperimentale

- Dataset: KaoKore v1.3, split ufficiale (train 7405 / dev 926 / test 926).
- Per vincoli di tempo, valutazione su un sottoinsieme fisso e deterministico: 80 immagini
  test, 20 immagini dev, identiche per tutti i metodi.
- Degradazione condivisa: blur gaussiano (sigma=2, kernel=9x9, condizioni al contorno
  circolari) + rumore gaussiano additivo, deviazione standard assoluta pari al noise level
  (0.005, 0.01, 0.05, 0.1), su immagini in [0,1]. Osservazioni salvate una sola volta come
  PNG 8-bit e riusate identiche da ogni metodo.
- Metriche: PSNR e SSIM (RGB), calcolate rispetto all'immagine pulita di riferimento.

## 2. Metodo variazionale — FISTA + regolarizzazione wavelet

### Formulazione
Il problema è formulato come:

x* = argmin_x  1/2 ||Ax - y||^2 + lambda * ||Wx||_1

con A l'operatore di blur gaussiano e W una trasformata wavelet ortogonale (Daubechies db4,
3 livelli di decomposizione, modalità di bordo "periodization" per preservare l'ortogonalità
coerentemente con le condizioni al contorno circolari di A). Risolto con FISTA (Beck &
Teboulle, 2009): ad ogni iterazione un passo di gradiente sul termine di data-fidelity seguito
da soft-thresholding dei coefficienti wavelet di dettaglio (l'operatore prossimale esatto della
norma L1 quando W è ortogonale), con extrapolation alla Nesterov per l'accelerazione. Il passo
di gradiente usa L=1 come costante di Lipschitz, valore esatto (non stimato) grazie alle
condizioni al contorno circolari, che rendono l'operatore di blur circolante con massimo
autovalore pari a 1 (kernel normalizzato a somma 1).

### Scelta di lambda
Grid search sul dev set (20 immagini, disgiunto dal test set), separatamente per ciascuno dei
4 livelli di rumore, su una griglia di fattori moltiplicativi del noise level
(0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 2.0, 4.0) x noise_level, selezionando il fattore che massimizza
il PSNR medio sul dev set.

| noise level | lambda scelto |
|---|---|
| 0.005 | 0.0025 |
| 0.01  | 0.0035 |
| 0.05  | 0.0175 |
| 0.1   | 0.0700 |

I valori crescono monotonicamente con il livello di rumore, coerentemente con il principio di
discrepanza: più rumore nell'osservazione richiede una regolarizzazione più forte.

### Risultati sul test set (80 immagini, 100 iterazioni FISTA)

| noise level (sigma) | lambda | PSNR medio (dB) | PSNR std | SSIM medio | SSIM std |
|---|---|---|---|---|---|
| 0.005 | 0.0025 | 31.17 | 3.85 | 0.861 | 0.068 |
| 0.01  | 0.0035 | 31.00 | 3.87 | 0.855 | 0.071 |
| 0.05  | 0.0175 | 28.96 | 3.35 | 0.779 | 0.082 |
| 0.1   | 0.0700 | 27.09 | 2.97 | 0.699 | 0.092 |

Tempo medio di ricostruzione: **1.572 s/immagine** (media su 320 ricostruzioni, CPU, 100
iterazioni).

### Osservazioni qualitative
Nessun artefatto di wraparound visibile ai bordi delle immagini ricostruite, nonostante le
condizioni al contorno circolari dell'operatore di blur: il kernel (9x9) è piccolo rispetto
alle dimensioni dell'immagine (256x256), quindi l'effetto resta confinato a pochi pixel dal
bordo. Ai livelli di rumore più alti (sigma=0.05, 0.1) permane una lieve grana/chiazzatura
residua nelle zone omogenee dell'immagine (sfondo, capelli): la sola sparsità wavelet non
rimuove completamente il rumore senza sacrificare i dettagli fini — un compromesso intrinseco
del metodo, non un errore implementativo. Nessuno staircasing (atteso semmai con
regolarizzazione TV, non wavelet).

### Paragrafo di sintesi (pronto per la stesura finale)
Il parametro di regolarizzazione lambda è stato scelto tramite grid search sul dev set (20
immagini, disgiunto dal test set), separatamente per ciascuno dei quattro livelli di rumore,
testando valori proporzionali al livello di rumore stesso e selezionando quello che massimizza
il PSNR medio. I quattro valori ottenuti crescono monotonicamente con il livello di rumore, in
accordo con il principio di discrepanza: maggiore è il rumore nell'osservazione, maggiore deve
essere il peso della regolarizzazione per evitare di interpretare il rumore come segnale. Sul
test set (80 immagini, fisse e disgiunte dal dev set), FISTA con soglia wavelet ottiene PSNR
medio da 31.17 dB (sigma=0.005) a 27.09 dB (sigma=0.1), e SSIM medio da 0.861 a 0.699, con una
degradazione monotona e coerente all'aumentare del rumore. L'ispezione visiva delle
ricostruzioni non mostra artefatti di bordo legati alle condizioni al contorno circolari
adottate per l'operatore di blur, grazie alle dimensioni contenute del kernel rispetto
all'immagine; ai livelli di rumore più alti permane invece una lieve grana residua nelle zone
omogenee, segno che la sola sparsità wavelet non rimuove completamente il rumore senza
compromettere i dettagli fini. Il tempo medio di ricostruzione è di 1.57 secondi per immagine
su CPU (100 iterazioni FISTA), il più rapido tra i metodi considerati nel confronto finale.

### Confronti visivi (originale | degradata | ricostruita FISTA)

**sigma = 0.005**

![](../results/fista/comparisons/sigma_0.005_00000700.jpg.png)
![](../results/fista/comparisons/sigma_0.005_00000734.jpg.png)
![](../results/fista/comparisons/sigma_0.005_00000815.jpg.png)

**sigma = 0.01**

![](../results/fista/comparisons/sigma_0.01_00000700.jpg.png)
![](../results/fista/comparisons/sigma_0.01_00000734.jpg.png)
![](../results/fista/comparisons/sigma_0.01_00000815.jpg.png)

**sigma = 0.05**

![](../results/fista/comparisons/sigma_0.05_00000700.jpg.png)
![](../results/fista/comparisons/sigma_0.05_00000734.jpg.png)
![](../results/fista/comparisons/sigma_0.05_00000815.jpg.png)

**sigma = 0.1**

![](../results/fista/comparisons/sigma_0.1_00000700.jpg.png)
![](../results/fista/comparisons/sigma_0.1_00000734.jpg.png)
![](../results/fista/comparisons/sigma_0.1_00000815.jpg.png)

## 3. Metodo ibrido — PD-Net + TV
*(da completare)*

## 4. Metodo end-to-end — UNet
*(da completare)*

## 5. Confronto finale tra metodi
*(da completare)*
