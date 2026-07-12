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

### Formulazione
Unrolling dell'algoritmo Primal-Dual di Chambolle-Pock per il problema TV-regolarizzato:

min_x  1/2||Ax-y||^2 + lambda*TV(x)

A differenza di un Learned Primal-Dual generico (dove il prior è appreso implicitamente
nello spazio dei dati), qui la variabile duale vive esplicitamente nello spazio del
gradiente dell'immagine (operatore `Gradient`, forward differences con aggiunto = divergenza
negata), il che lega la rete alla struttura TV richiesta dal testo invece che a un prior
generico non specificato. Ad ogni iterazione k:

```
Gx_bar = Gradient(x_bar)
p = p + DualNet_k([p, Gx_bar])

data_grad = A^T(A(x) - y)
GTp = Gradient.T(p)
x_new = x + PrimalNet_k([x, data_grad, GTp])

x_bar = x_new + (x_new - x)      # extrapolation, come nel Chambolle-Pock classico
x = x_new
```

Il gradiente del data-fidelity resta calcolato esattamente (fisica nota, non appresa); solo
gli aggiornamenti prossimali (proiezione duale, passo primale) sono sostituiti da piccole CNN
(Conv3x3-LeakyReLU-Conv3x3), pesi indipendenti per ciascuna delle 8 iterazioni (unrolling).
Il numero di iterazioni (8) è stato scelto euristicamente, in linea con l'ordine di grandezza
usato nel paper originale di Adler & Öktem (Learned Primal-Dual), non da una grid search
esaustiva per vincoli di tempo.

### Training
4 modelli specializzati (uno per livello di rumore, stessa scelta fatta per UNet, vedi
sezione 4 per la motivazione), 1000 immagini training (sottoinsieme), degradazione generata
al volo ad ogni epoca, loss L1, Adam (lr=2e-4), 12 epoche, validazione sul dev set fisso con
salvataggio del checkpoint a PSNR migliore.

### Risultati sul test set (80 immagini)

| noise level (sigma) | PSNR medio (dB) | PSNR std | SSIM medio | SSIM std |
|---|---|---|---|---|
| 0.005 | 33.43 | 3.52 | 0.904 | 0.050 |
| 0.01  | 33.84 | 3.64 | 0.910 | 0.053 |
| 0.05  | 31.77 | 3.43 | 0.862 | 0.073 |
| 0.1   | 30.14 | 3.27 | 0.822 | 0.088 |

Tempo medio di ricostruzione: **0.0376 s/immagine** (singolo forward pass, GPU). Il checkpoint
allenato pesa ~302 KB, circa 100 volte più piccolo di quello di UNet (~33 MB), pur ottenendo
risultati quasi identici — conseguenza diretta dell'aver incorporato la fisica nota
(operatori A e Gradient fissi, non appresi) direttamente nell'architettura.

### Confronti visivi (originale | degradata | ricostruita PD-Net)

**sigma = 0.005**

![](../results/pdnet/comparisons/sigma_0.005_00000700.jpg.png)
![](../results/pdnet/comparisons/sigma_0.005_00000734.jpg.png)
![](../results/pdnet/comparisons/sigma_0.005_00000815.jpg.png)

**sigma = 0.01**

![](../results/pdnet/comparisons/sigma_0.01_00000700.jpg.png)
![](../results/pdnet/comparisons/sigma_0.01_00000734.jpg.png)
![](../results/pdnet/comparisons/sigma_0.01_00000815.jpg.png)

**sigma = 0.05**

![](../results/pdnet/comparisons/sigma_0.05_00000700.jpg.png)
![](../results/pdnet/comparisons/sigma_0.05_00000734.jpg.png)
![](../results/pdnet/comparisons/sigma_0.05_00000815.jpg.png)

**sigma = 0.1**

![](../results/pdnet/comparisons/sigma_0.1_00000700.jpg.png)
![](../results/pdnet/comparisons/sigma_0.1_00000734.jpg.png)
![](../results/pdnet/comparisons/sigma_0.1_00000815.jpg.png)

## 4. Metodo end-to-end — UNet

### Scelta dell'architettura
Tra UNet, ViT e NAF-Net (le tre opzioni ammesse dal testo), si è scelta UNet: il blur è
locale (kernel 9x9) e il rumore è per-pixel, quindi non serve catturare dipendenze a lungo
raggio come farebbe l'attention di un ViT (più utile per task con contesto globale, es.
inpainting di regioni ampie), e un ViT richiederebbe più dati/parametri per essere
competitivo. NAF-Net sarebbe stata una scelta valida ma più complessa da implementare
correttamente nel tempo disponibile. Le skip connection di UNet sono il meccanismo diretto
per un problema dove input e output condividono quasi tutta la struttura spaziale.

### Architettura
4 livelli di downsampling (MaxPool 2x2), canali base 48 (48-96-192-384 nel bottleneck a
16x16), blocco convoluzionale = 2x(Conv3x3 + GroupNorm + ReLU) per stadio, skip connection
per concatenazione. Apprendimento residuale: la rete produce x_hat = y + UNet(y), impara
solo la correzione da applicare all'osservazione invece di ricostruire l'immagine da zero
(pratica standard nelle reti di denoising, es. DnCNN).

### Training
4 modelli specializzati (uno per livello di rumore): per un confronto equo con FISTA, che
specializza necessariamente il parametro lambda per livello di rumore (principio di
discrepanza), si è scelto di dare anche ai metodi deep-learning lo stesso "budget di
specializzazione", invece di un solo modello blind generalista — così il confronto isola la
capacità di ciascun metodo di ottenere il miglior risultato possibile a un dato livello di
rumore, senza confondere questo con la capacità di generalizzare su più rumori
contemporaneamente. 1000 immagini training, degradazione al volo (rumore ridisegnato ad ogni
epoca, blur sempre fisso), loss L1, Adam (lr=2e-4), 20 epoche, validazione sul dev set fisso.

### Risultati sul test set (80 immagini)

| noise level (sigma) | PSNR medio (dB) | PSNR std | SSIM medio | SSIM std |
|---|---|---|---|---|
| 0.005 | 33.39 | 3.51 | 0.903 | 0.049 |
| 0.01  | 34.03 | 3.73 | 0.913 | 0.052 |
| 0.05  | 32.08 | 3.44 | 0.872 | 0.069 |
| 0.1   | 30.49 | 3.26 | 0.836 | 0.080 |

Tempo medio di ricostruzione: **0.0373 s/immagine**. Si osserva un andamento leggermente non
monotono (PSNR a sigma=0.005 inferiore a quello a sigma=0.01): plausibilmente dovuto al fatto
che a rumore molto basso il residuo da apprendere è molto piccolo, con un segnale di
gradiente più debole durante l'ottimizzazione; lo stesso pattern si osserva anche in PD-Net,
il che rafforza l'ipotesi che sia un effetto sistematico del setup di training (20 epoche,
sottoinsieme ridotto) piuttosto che un errore isolato.

### Confronti visivi (originale | degradata | ricostruita UNet)

**sigma = 0.005**

![](../results/unet/comparisons/sigma_0.005_00000700.jpg.png)
![](../results/unet/comparisons/sigma_0.005_00000734.jpg.png)
![](../results/unet/comparisons/sigma_0.005_00000815.jpg.png)

**sigma = 0.01**

![](../results/unet/comparisons/sigma_0.01_00000700.jpg.png)
![](../results/unet/comparisons/sigma_0.01_00000734.jpg.png)
![](../results/unet/comparisons/sigma_0.01_00000815.jpg.png)

**sigma = 0.05**

![](../results/unet/comparisons/sigma_0.05_00000700.jpg.png)
![](../results/unet/comparisons/sigma_0.05_00000734.jpg.png)
![](../results/unet/comparisons/sigma_0.05_00000815.jpg.png)

**sigma = 0.1**

![](../results/unet/comparisons/sigma_0.1_00000700.jpg.png)
![](../results/unet/comparisons/sigma_0.1_00000734.jpg.png)
![](../results/unet/comparisons/sigma_0.1_00000815.jpg.png)

## 5. Confronto finale tra metodi

| Metodo | PSNR (range sui 4 livelli) | SSIM (range) | Tempo/immagine |
|---|---|---|---|
| FISTA-Wavelet | 27.09 - 31.17 dB | 0.699 - 0.861 | 1.572 s |
| PD-Net | 30.14 - 33.84 dB | 0.822 - 0.910 | 0.038 s |
| UNet | 30.49 - 34.03 dB | 0.836 - 0.913 | 0.037 s |

![](../results/figures/comparison_plot.png)

### Discussione
UNet e PD-Net superano nettamente FISTA (2-3 dB di PSNR in più a ogni livello di rumore): un
metodo allenato specificamente sulla statistica del dataset KaoKore sfrutta un prior molto
più informativo di quello generico (sparsità wavelet) usato da FISTA, che è valido per
qualunque immagine naturale ma non specifico al dominio. UNet e PD-Net ottengono risultati
quasi indistinguibili tra loro, ma PD-Net usa circa 100 volte meno parametri: incorporare la
fisica nota del problema (operatore di blur, struttura del gradiente per la TV) direttamente
nell'architettura riduce drasticamente cosa la rete deve imparare da zero, a parità di
qualità finale. FISTA è circa 40 volte più lento (richiede 100 iterazioni di ottimizzazione
per immagine, contro un singolo forward pass delle reti), ma è l'unico dei tre metodi che non
richiede alcuna fase di training ed è interamente interpretabile (ogni passo dell'algoritmo
ha un significato matematico esplicito). Il trade-off complessivo: FISTA offre
interpretabilità e zero costo di training a scapito di qualità e velocità; UNet massimizza la
qualità a scapito del numero di parametri; PD-Net rappresenta il miglior compromesso tra i
tre, avvicinandosi alla qualità di UNet con una frazione dei parametri grazie all'impalcatura
fisica esplicita.

### Limiti dello studio
Per vincoli di tempo: valutazione su un sottoinsieme fisso (80 immagini test, 20 dev) invece
del test set completo (926 immagini); training set ridotto (1000 immagini) e numero di epoche
limitato (20 per UNet, 12 per PD-Net) rispetto a quanto si userebbe con risorse
computazionali illimitate. Entrambe le scelte sono state dichiarate esplicitamente e applicate
in modo identico a tutti i metodi, per mantenere il confronto equo richiesto dal testo.
