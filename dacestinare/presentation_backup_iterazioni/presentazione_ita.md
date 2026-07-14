# Presentazione — Deblur & Denoise su KaoKore come problema inverso

Trascrizione testuale di `presentazione_ita.pptx`, slide per slide. Utile per ripassare
o parlare a braccio senza dover aprire PowerPoint.

---

## Slide 1 — Titolo

**Deblur & Denoise su KaoKore come problema inverso**

Computational Imaging 2025-26
Gabriele Centonze

---

## Slide 2 — Descrizione del progetto

- Task: deblur + denoise, formulato come problema inverso y = Ax + n
  - A: blur gaussiano (sigma=2, kernel 9x9); n: rumore gaussiano additivo, 4 livelli (0.005, 0.01, 0.05, 0.1)
- Dataset: KaoKore v1.3, ritratti RGB 256x256 (9257 immagini)
  - Split ufficiale train/dev/test fornito nel labels.csv del dataset: 7405 / 926 / 926
- Obiettivo: implementare e confrontare criticamente metodi di famiglie diverse
  - sotto le stesse condizioni sperimentali (stessi input degradati per ogni metodo)
- 3 dei 4 metodi proposti (opzione consentita per progetti da uno studente):
  - Variazionale (FISTA + Wavelet), Ibrido (PD-Net + TV), End-to-end (UNet)

---

## Slide 3 — Metodologia: panoramica

- Il problema inverso è mal posto: A attenua le alte frequenze spaziali
  - l'inversione ingenua amplifica il rumore -> serve regolarizzazione / un prior
- Tre modi diversi di introdurre questo prior:
  - Variazionale: un prior esplicito e scelto a mano (sparsità wavelet), nessun training
  - Ibrido: fisica nota (A, gradiente dell'immagine) + poche componenti apprese
  - End-to-end: l'intera mappa y -> x è appresa dai dati
- Tutti e tre condividono lo stesso operatore di degradazione e leggono
  - esattamente gli stessi file PNG degradati in valutazione (confronto equo)

---

## Slide 4 — Metodologia: il filo conduttore *(slide discorsiva)*

Tutti e tre i metodi rispondono alla stessa domanda di fondo: dato che A distrugge
informazione (attenua le alte frequenze), come iniettiamo abbastanza conoscenza a
priori su cosa sia un'immagine plausibile per colmare quel vuoto, senza amplificare
il rumore al posto del segnale?

Quello che cambia tra i tre metodi è quanto di quel prior sia scelto a mano oppure
appreso dai dati. FISTA sta a un estremo: il prior è interamente costruito a mano
(le immagini naturali sono sparse in base wavelet), e l'unica cosa tarata è uno
scalare per livello di rumore. UNet sta all'estremo opposto: nessuna informazione
sulla fisica è data alla rete — tutto, inclusa una nozione implicita di come sia un
ritratto KaoKore degradato, viene dedotto solo dagli esempi di training.

PD-Net sta deliberatamente nel mezzo: la fisica già nota esattamente (operatore A,
gradiente dell'immagine) resta fissa dentro la rete, e solo i passi senza forma
chiusa (gli aggiornamenti prossimali di Chambolle-Pock) sono appresi. Per questo
raggiunge l'accuratezza di UNet con due ordini di grandezza in meno di parametri —
non deve mai imparare ciò che la fisica gli dà già gratis.

---

## Slide 5 — Metodologia: Variazionale (FISTA + Wavelet)

**Formula:**
```
min_x  1/2||Ax-y||²  +  λ||Wx||₁
```
(W: trasformata wavelet ortogonale)

- Convesso ma non liscio -> metodo proximal gradient (Beck & Teboulle, 2009)
- Per W ortogonale, il prossimale del termine L1 = soft-thresholding dei coefficienti wavelet
- Estrapolazione alla Nesterov: velocità di convergenza O(1/k²) invece di O(1/k)
- Passo 1/L: L = ||A^T A|| = 1 esattamente, grazie alle condizioni al contorno circolari
  - (A diventa un operatore circolante simmetrico, diagonalizzato dalla trasformata di Fourier)

---

## Slide 6 — Metodologia: Ibrido (PD-Net + TV)

**Formula:**
```
min_x  1/2||Ax-y||²  +  λ·TV(x)
```

- Unrolling dell'algoritmo primal-dual di Chambolle-Pock
- I passi prossimali/di proiezione esatti sono sostituiti da piccole CNN apprese
- La variabile duale vive nello spazio del gradiente (non dei dati): lega la rete
  - esplicitamente alla struttura TV, a differenza di un primal-dual generico appreso
- Il gradiente del data-fidelity A^T(Ax-y) resta esatto (non appreso) ad ogni iterazione
  - -> ibrido: fisica nota + correzione appresa, non una scatola nera

---

## Slide 7 — Metodologia: End-to-end (UNet)

- Completamente supervisionato: impara la mappa diretta y -> x dalle coppie di training
- Scelta rispetto a ViT / NAF-Net: la degradazione è spazialmente locale (kernel piccolo,
  - rumore per-pixel) -> non serve attention a lungo raggio; più semplice da implementare
  - bene nel tempo disponibile
- Encoder-decoder con skip connection: preservano il dettaglio spaziale fine
  - attraverso il collo di bottiglia, necessario perché input e output condividono la struttura
- Apprendimento residuale (la rete impara solo la correzione):

**Formula:**
```
x̂ = y + UNet(y)
```

---

## Slide 8 — Implementazione: FISTA + Wavelet

- Wavelet: Daubechies db4, 3 livelli di decomposizione, mode='periodization'
  - (periodization mantiene la DWT esattamente ortogonale, necessario perché il prossimale sia esatto)
- 100 iterazioni per immagine, nessun parametro appreso
- lambda tarato con grid search sul dev set (20 immagini), separatamente per livello di rumore
  - (0.005 -> 0.0025, 0.01 -> 0.0035, 0.05 -> 0.0175, 0.1 -> 0.07: cresce con il rumore,
  - come atteso dal principio di discrepanza)

---

## Slide 9 — Implementazione: PD-Net

*(Diagramma architetturale: schema del ciclo primale/duale unrolled, blocchi DualNet_k e PrimalNet_k)*

- 8 iterazioni di unrolling, pesi indipendenti per iterazione; blocchi CNN: Conv3x3-LeakyReLU-Conv3x3
- 4 modelli specializzati (uno per livello di rumore), 700 immagini di training, 12 epoche, Adam (lr=2e-4), loss L1

---

## Slide 10 — Implementazione: UNet

*(Diagramma architetturale: encoder-decoder a 4 livelli con skip connection)*

- 4 livelli di downsampling, canali base 48, GroupNorm (stabile con batch size piccoli)
- 4 modelli specializzati (uno per livello di rumore), 1000 immagini di training, 20 epoche, Adam (lr=2e-4), loss L1

---

## Slide 11 — Introduzione agli esperimenti

- Degradazione: blur gaussiano sigma=2, kernel 9x9, condizioni al contorno circolari
- Rumore: gaussiano additivo, std = livello di rumore direttamente (0.005/0.01/0.05/0.1), su immagini in [0,1]
- Osservazioni degradate generate una volta, salvate come PNG a 8 bit -> input identici per tutti i metodi
- Scelte di scope dettate dal tempo (dichiarate esplicitamente, non nascoste):
  - valutazione su un sottoinsieme fisso (80 test + 20 dev) invece delle 926/926 totali
  - 3 metodi su 4 (consentito per progetti da uno studente); scartato il metodo generativo (Diffusion+DPS)
  - 4 modelli specializzati per metodo appreso, per un confronto equo con il lambda per livello di FISTA

---

## Slide 12 — Risultati numerici: tabella riassuntiva

*(Tabella con colonne: Metodo, Livello di rumore, PSNR (dB), SSIM, Tempo/immagine (s) — dati da `results/tables/comparison_summary.csv`)*

| Metodo | σ=0.005 | σ=0.01 | σ=0.05 | σ=0.1 | Tempo/immagine |
|---|---|---|---|---|---|
| FISTA-Wavelet | 31.17 dB | 31.00 dB | 28.96 dB | 27.09 dB | 1.572 s |
| PD-Net | 33.43 dB | 33.84 dB | 31.77 dB | 30.14 dB | 0.038 s |
| UNet | 33.39 dB | 34.03 dB | 32.08 dB | 30.49 dB | 0.037 s |

---

## Slide 13 — Risultati numerici: plot comparativo

*(Immagine: `results/figures/comparison_plot.png` — PSNR e SSIM vs livello di rumore, scala log, una curva per metodo)*

---

## Slide 14 — Risultati numerici: confronto visivo

*(Immagine: `results/figures/composite_comparison.png` — stessa immagine test, 4 righe (una per livello di rumore) × 5 colonne: originale, degradata, FISTA, PD-Net, UNet)*

---

## Slide 15 — Come leggere l'anomalia non monotona *(slide discorsiva)*

Un risultato merita uno sguardo più attento invece di essere riportato e basta: sia
UNet che PD-Net raggiungono un PSNR leggermente più basso a sigma=0.005 che a
sigma=0.01 — il caso più facile, meno rumoroso, va peggio. Vale la pena chiedersi
perché, non solo constatarlo.

Entrambi i metodi usano apprendimento residuale: l'output è y più una correzione
appresa. A rumore molto basso quella correzione è minuscola, quindi il segnale di
training che guida l'ottimizzazione è proporzionalmente più debole che a sigma=0.01,
dove la correzione è più grande e più facile da imparare nello stesso numero fisso
di epoche.

Lo stesso pattern compare indipendentemente in due architetture diverse, allenate
separatamente — questo è ciò che lo rende un risultato genuino e non rumore di un
singolo run: indica una proprietà del setup di training, non una coincidenza
dell'inizializzazione. Più epoche al livello di rumore più basso probabilmente
chiuderebbero questo divario.

---

## Slide 16 — Cosa ci dice l'efficienza di PD-Net *(slide discorsiva)*

PD-Net eguaglia l'accuratezza di UNet ad ogni livello di rumore con ~100 volte meno
parametri (302 KB contro 33 MB). Questo è l'argomento empirico centrale a favore
delle architetture ibride fisicamente informate, non un dettaglio implementativo
minore.

Una rete puramente data-driven deve scoprire, solo dagli esempi, che disfare una
convoluzione nota fa parte del compito — parte dei suoi parametri e dei dati di
training sono spesi a ri-derivare qualcosa che era già noto prima ancora di
iniziare il training.

PD-Net riceve A e il gradiente dell'immagine come blocchi fissi ed esatti; le sue
componenti apprese riempiono solo la parte senza forma chiusa. In termini di machine
learning: un forte inductive bias sostituisce dati e parametri — e ci si aspetta
generalizzi meglio fuori dalle condizioni esatte di training, perché parte del suo
comportamento è garantito, non appreso.

---

## Slide 17 — Sull'onestà nel dichiarare i nostri limiti *(slide discorsiva)*

Questo progetto è stato svolto con un vincolo di tempo reale e fisso, e diverse
scelte di scope sono state fatte esplicitamente per renderlo possibile: un
sottoinsieme test di 80 immagini invece di 926, tre dei quattro metodi proposti,
un training set e un numero di epoche ridotti.

Niente di questo è nascosto dentro i numeri — è dichiarato apertamente nel report
e nella documentazione del codice, perché un risultato con limiti noti è più
affidabile di uno che sembra completo ma nasconde scorciatoie.

Allenare silenziosamente su meno immagini senza dirlo avrebbe prodotto numeri più
lucidi in apparenza, ma non avrebbe retto a una domanda diretta sulla metodologia.
Dichiarare i limiti è ciò che rende ogni numero di questa presentazione difendibile.

---

## Slide 18 — Conclusioni

- UNet e PD-Net superano FISTA di 2-3 dB di PSNR ad ogni livello di rumore:
  - un prior appreso sulla statistica di KaoKore batte un prior generico di sparsità wavelet
- UNet e PD-Net ottengono accuratezza quasi identica, ma PD-Net usa ~100 volte meno parametri
  - (checkpoint 302 KB vs 33 MB): incorporare la fisica nota riduce cosa la rete deve imparare
- FISTA è ~40 volte più lento per immagine (ottimizzazione iterativa vs un solo forward pass),
  - ma non richiede training ed è completamente interpretabile, a differenza dei due metodi appresi
- Limiti: sottoinsieme di valutazione e training set/epoche ridotti, per vincoli di tempo
- Sviluppi futuri: test set e training set completi, più epoche, aggiungere il 4° metodo
  - (Diffusion + DPS), provare NAF-Net come architettura end-to-end alternativa
