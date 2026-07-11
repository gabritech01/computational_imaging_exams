# Piano di Sviluppo — Progetto Computational Imaging
## Deblur & Denoise su KaoKore 256x256 come problema inverso

Studente singolo — 4 metodi implementati (variazionale, end-to-end, generativo, ibrido).
Compute: Google Colab / Colab Pro.

---

## 1. Obiettivo e vincoli del progetto (dal testo d'esame)

- **Task**: inverse problem `y = A x + n` con `A` = blur gaussiano (sigma=2, kernel=9), `n` = rumore
  gaussiano additivo a 4 livelli: 0.005, 0.01, 0.05, 0.1.
- **Dataset**: KaoKore, resize a 256x256, split train/val/test documentato e giustificato.
- **Metodi richiesti** (tutti e 4, come deciso):
  1. **Variazionale**: FISTA + regolarizzazione wavelet (sparsity in dominio wavelet).
  2. **End-to-end**: **UNet** supervisionato (input degradato → output ricostruito).
  3. **Generativo**: Diffusion model + **DPS** (Diffusion Posterior Sampling) adattato al task,
     a risoluzione **64x64** (kernel blur ridotto a 3, come da testo per il caso ridimensionato).
  4. **Ibrido**: **PD-Net** (Primal-Dual Network, unrolling di Chambolle-Pock) con regolarizzazione
     TV, numero di iterazioni di unrolling da definire empiricamente.
- **Vincolo critico**: tutti i metodi devono essere valutati sugli **stessi input degradati**
  (stesso seed di rumore per immagine/livello, stesso kernel di blur).
- **Metriche**: PSNR, SSIM su test set, per ciascun livello di rumore, per ciascun metodo.
- **Deliverable finali**: report scritto, presentazione (PowerPoint/Beamer, ~45 min, template
  fornito), codice documentato (GitHub, link da fornire — **ma vedi nota sotto**), tutti i
  risultati precalcolati e inclusi nella presentazione (non si può eseguire codice all'esame).

> **Nota da chiarire con i docenti**: il testo del progetto dice "possibly pushed on Github",
> ma le slide "Remainders" dicono "The project must NOT be uploaded anywhere". Trattarlo come:
> repo **privato** su GitHub (non pubblico), da mostrare/linkare solo ai docenti se richiesto.
> Da verificare via email prima della consegna.

---

## 2. Struttura del repository

```
Progetto-Deblur-Denoise-KaoKore/
├── PIANO_DI_SVILUPPO.md
├── README.md
├── requirements.txt / environment.yml
├── data/                          # dataset scaricato (gitignored, non su GitHub)
├── notebooks/                     # notebook Colab, uno per fase/metodo
│   ├── 00_dataset_exploration.ipynb
│   ├── 01_preprocessing_and_degradation.ipynb
│   ├── 02_variational_fista_wavelet.ipynb
│   ├── 03_unet_end2end.ipynb
│   ├── 04_diffusion_dps.ipynb
│   ├── 05_pdnet_hybrid.ipynb
│   └── 06_comparison_and_plots.ipynb
├── src/
│   ├── data/                      # dataset class, split, KaoKore loader
│   ├── degradation/                # operatore di blur+noise condiviso, forward operator A
│   ├── methods/
│   │   ├── variational_fista/     # FISTA + wavelet shrinkage
│   │   ├── end2end_unet/          # architettura UNet, training loop
│   │   ├── generative_dps/        # DDPM/score model + DPS sampler
│   │   └── hybrid_pdnet/          # PD-Net unrolled, TV prox
│   ├── eval/                      # PSNR/SSIM, comparison plot generator
│   └── utils/                     # seeding, IO, visualizzazione
├── results/
│   ├── figures/                   # immagini per la presentazione
│   ├── tables/                    # CSV con metriche
│   └── checkpoints/                # pesi modelli (grandi, gitignored)
├── report/                        # report scritto (PDF/LaTeX o Markdown)
└── presentation/                  # slide finali (basate sul template fornito)
```

**Principio chiave**: il **degradation operator** (blur+noise) e il **set di immagini test degradate**
sono generati **una sola volta**, salvati su disco con seed fissato, e riusati identici da tutti
e 4 i metodi. Questo garantisce il confronto equo richiesto dal testo.

---

## 3. Fasi di sviluppo

### Fase 0 — Setup e organizzazione (0.5 giorni)
- [ ] Creare repo Git locale + remoto privato su GitHub.
- [ ] Setup ambiente Colab (requirements.txt: torch, numpy, scipy, scikit-image, pywavelets, matplotlib).
- [ ] Script di download/mount KaoKore su Google Drive (per persistenza tra sessioni Colab).
- [ ] Definire `src/utils/seed.py` per riproducibilità totale (seed unico per tutto il progetto).

### Fase 1 — Dataset e preprocessing (1 giorno)
- [ ] Scaricare KaoKore 256x256, ispezionare struttura (classi: gender/status, split ufficiali).
- [ ] Definire split train/val/test (usare split ufficiale KaoKore se disponibile, altrimenti
      es. 80/10/10 stratificato) — **documentare e giustificare la scelta nel report**.
- [ ] Normalizzazione in [0,1] o [-1,1] (coerente con range di training delle reti scelte).
- [ ] Verificare dimensioni reali immagini KaoKore; resize a 256x256 se necessario (richiesto dal testo).
- [ ] `src/data/kaokore_dataset.py`: PyTorch `Dataset` con caching per velocità su Colab.

### Fase 2 — Operatore di degradazione condiviso (1 giorno)
- [ ] `src/degradation/blur.py`: kernel gaussiano 2D (sigma=2, size=9), applicato via convoluzione
      (padding riflettente per evitare artefatti ai bordi).
- [ ] `src/degradation/noise.py`: rumore gaussiano additivo, 4 livelli (0.005/0.01/0.05/0.1).
- [ ] `src/degradation/generate_degraded_set.py`: genera e salva su disco (una volta sola, seed
      fissato) le versioni degradate di validation e test set, per ciascun livello di rumore →
      questi file sono l'input **immutabile** condiviso da tutti i metodi.
- [ ] Variante a 64x64 con kernel size=3 per il metodo generativo (Fase 2 bis, stesso principio).
- [ ] Sanity check: visualizzare coppie (ground truth, blur, blur+noise) per ogni livello.

### Fase 3 — Metodo Variazionale: FISTA + Wavelet (2-3 giorni)
- [ ] Formulazione: `min_x 1/2||Ax-y||^2 + lambda * ||W x||_1` (W = trasformata wavelet, es. Daubechies).
- [ ] Implementare FISTA (Beck & Teboulle 2009): step size da Lipschitz di A^T A, momentum di Nesterov,
      soft-thresholding nel dominio wavelet ad ogni iterazione.
- [ ] Calcolo Lipschitz constant di A (operatore di blur, tramite norma dell'operatore o power iteration).
- [ ] Selezione euristica di lambda (grid search su validation set, per ciascun livello di rumore
      — atteso lambda crescente con il livello di rumore).
- [ ] Criterio di stop (numero iterazioni fisso o tolleranza su variazione obiettivo).
- [ ] Output: immagini ricostruite test set, metriche PSNR/SSIM, curva di convergenza (obiettivo vs iterazioni).

### Fase 4 — Metodo End-to-End: UNet (3-4 giorni, training su Colab)
- [ ] Architettura UNet: encoder-decoder con skip connections, adattata a input/output 256x256x3.
      Giustificare profondità e numero di canali in base a task (deblur+denoise, non serve
      down-sampling estremo — 4-5 livelli sufficienti).
- [ ] Input: immagine degradata (a un livello di rumore, o multi-livello con conditioning sul
      livello — **decisione da esplicitare nel report**: training singolo generalista vs.
      training per-livello). Raccomandato: training singolo con rumore variabile in training
      per generalizzare meglio, poi valutazione sui 4 livelli fissi.
- [ ] Loss: combinazione L1/L2 + eventualmente SSIM loss per allineare meglio con la metrica finale.
- [ ] Training: data augmentation leggera (flip), batch size compatibile con GPU Colab (T4/A100),
      early stopping su validation loss, checkpoint salvati su Drive (Colab si disconnette).
- [ ] Log training (loss curves) da includere nella presentazione.

### Fase 5 — Metodo Generativo: Diffusion + DPS (4-5 giorni, il più oneroso)
- [ ] Training di un modello di diffusione (DDPM semplificato) **non condizionato** sulle immagini
      KaoKore a 64x64 (dataset ridotto in dimensione per contenere tempi di training su Colab).
- [ ] Architettura: UNet piccola per la stima del rumore (score/epsilon-predictor), noise schedule
      lineare o cosine, T ridotto (es. 200-500 step) per contenere training/inference time.
- [ ] Implementare **DPS** (Chung et al. 2022): ad ogni step di reverse diffusion, correggere la
      stima con il gradiente della log-likelihood dei dati misurati `y` rispetto a `A`, con step
      size adattivo (`zeta_t` proporzionale a `1/||y - A x_hat||`).
- [ ] Adattare DPS al forward operator di blur+noise (non è inpainting/CT, quindi A è la convoluzione
      col kernel gaussiano — gradiente calcolabile in forma chiusa via convoluzione trasposta).
- [ ] Iperparametri da tarare euristicamente: numero di step di sampling, guidance scale (zeta),
      eventuale early-stopping del numero di step di reverse diffusion — documentare il processo
      di tuning (griglia di prova + criterio di scelta) per la discussione orale.
- [ ] Valutare su test set a 64x64 con kernel size=3, 4 livelli di rumore (adattati in scala se
      necessario e giustificato).

### Fase 6 — Metodo Ibrido: PD-Net + TV (2-3 giorni)
- [ ] Formulazione: unrolling dell'algoritmo Primal-Dual di Chambolle-Pock per
      `min_x 1/2||Ax-y||^2 + lambda * TV(x)`, sostituendo il prossimale TV esatto con un blocco
      neurale learnable ad ogni iterazione (à la Adler & Öktem, "Learned Primal-Dual").
- [ ] Numero di iterazioni di unrolling: scegliere un valore (es. 5-10) come compromesso
      qualità/costo computazionale — **giustificare la scelta empiricamente** (curva PSNR vs
      numero di iterazioni sul validation set).
- [ ] Training end-to-end dei blocchi CNN del PD-Net (piccole reti conv per ogni step primal/dual),
      con l'operatore A e A^T (blur/blur trasposto) fissi e non allenati.
- [ ] Confrontare esplicitamente con il TV puro (senza componente neurale) come baseline interna,
      per mostrare il valore aggiunto della componente ibrida — utile per la discussione.

### Fase 7 — Valutazione comparativa e produzione risultati (2 giorni)
- [ ] `src/eval/metrics.py`: PSNR e SSIM standardizzati (stessa libreria/formula per tutti i metodi).
- [ ] Tabella riassuntiva: righe = metodi, colonne = livelli di rumore, celle = PSNR/SSIM medi ± std
      sul test set. Salvata in `results/tables/`.
- [ ] Plot comparativo: PSNR (e SSIM) vs livello di rumore, una curva per metodo (il "comparison
      plot within all methods" richiesto esplicitamente dal testo).
- [ ] Selezione di crop/immagini rappresentative (non tutto il test set) + eventuali difference
      images (output − ground truth) per evidenziare artefatti caratteristici di ciascun metodo
      (es. over-smoothing variazionale/UNet a rumore alto, allucinazioni del generativo, ecc.).
- [ ] Scrivere discussione critica: punti di forza/debolezza di ciascuna famiglia di metodi,
      rispetto a tempo di inferenza, stabilità, generalizzazione, qualità percettiva vs fedeltà.

### Fase 8 — Report scritto (1-2 giorni)
- [ ] Sezioni: Introduzione/task, Dataset e preprocessing, Metodologia (4 metodi), Setup
      sperimentale (parametri, hardware, tempi), Risultati (tabelle/plot), Discussione, Conclusioni,
      Bibliografia.
- [ ] Includere tutte le scelte di iperparametri con relativa giustificazione (richiesto esplicitamente).

### Fase 9 — Presentazione (segue il template ufficiale) (1-2 giorni)
Slide da produrre, mappate 1:1 sul template "trace for the presentation.pdf":
1. Titolo (nome, progetto, data).
2. **Description of the project** — task, dataset, formulazione come inverse problem.
3. **Methodology** — breve teoria per ciascuno dei 4 metodi (FISTA/wavelet, UNet, DPS, PD-Net/TV).
4. **Implementation** — architetture (schema grafico per UNet/Diffusion/PD-Net), loss, iperparametri
   di training, dataset/split.
5. **Introduce the experiments** — dettaglio blur kernel, livelli di rumore, risoluzioni usate
   per il metodo generativo.
6. **Presentation of numerical results** — tabelle PSNR/SSIM, comparison plot, immagini/crop
   selezionati, difference images.
7. **Conclusions** — sintesi, limiti, sviluppi futuri.
8. **Bibliography** — FISTA (Beck&Teboulle), DPS (Chung et al.), Learned Primal-Dual
   (Adler&Öktem), UNet (Ronneberger et al.), KaoKore paper.
- [ ] **Tutti i risultati devono essere precalcolati e incorporati come immagini/tabelle statiche**
      nelle slide (non si può eseguire codice durante l'esame).
- [ ] Provare il timing: ~45 minuti totali, distribuire tempo tra sezioni (indicativamente:
      5' intro, 10' metodologia, 10' implementazione, 15' risultati, 5' conclusioni).

### Fase 10 — Rifinitura codice e documentazione (1 giorno)
- [ ] README.md con istruzioni di riproducibilità (setup, come rigenerare dati degradati,
      come lanciare training/eval per ciascun metodo).
- [ ] Docstring essenziali su moduli chiave (operatore di degradazione, FISTA, DPS sampler, PD-Net).
- [ ] Pulizia notebook (rimuovere celle di debug, output ridondanti).
- [ ] Repo GitHub privato, commit history pulita.

---

## 4. Stima tempi totale

~18-22 giorni di lavoro effettivo (part-time), la parte più rischiosa in termini di tempo è la
**Fase 5 (Diffusion+DPS)** per via dei limiti di sessione Colab — mitigata scegliendo risoluzione
64x64 e T ridotto, con checkpoint frequenti su Google Drive.

## 5. Rischi principali e mitigazioni
- **Disconnessioni Colab durante training lunghi** → checkpoint ogni N epoche su Drive, script di
  resume automatico.
- **DPS instabile/lento** → partire da un DDPM piccolo e ben validato su ricostruzione unconditional
  prima di aggiungere la guidance DPS; tarare zeta_t con pochi esempi prima del run completo.
- **Confronto non equo tra metodi** → operatore di degradazione e immagini di test generate una
  sola volta e riusate identiche (Fase 2), verificato con checksum/hash dei file degradati.
- **Tempo per il report/slide sottostimato** → bloccare le Fasi 7-9 con almeno 4-5 giorni dedicati
  prima della data d'esame.

---

## 6. Prossimi passi

Una volta approvato questo piano:
1. Avvio Fase 0 e Fase 1 (setup + dataset) — implementazione affidata a un agente Opus per
   massimizzare qualità del codice.
2. Procedo metodo per metodo (Fasi 3→6), con checkpoint di revisione dopo ciascun metodo.
3. Fasi 7-9 (valutazione, report, slide) a valle di tutti i metodi.
