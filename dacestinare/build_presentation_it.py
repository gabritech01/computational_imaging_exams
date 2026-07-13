"""Versione italiana della presentazione, stessa struttura e funzioni di supporto di
build_presentation.py, solo i contenuti testuali cambiano."""
from pptx.util import Inches

from dacestinare.build_presentation import (
    new_pres, add_title_slide, add_content_slide, add_bullets, add_image, add_table,
    add_prose, read_comparison_csv,
)


def build():
    prs = new_pres()

    # 1. Titolo
    add_title_slide(
        prs,
        "Deblur & Denoise su KaoKore\ncome problema inverso",
        ["Computational Imaging 2025-26", "Gabriele Centonze"],
    )

    # 2. Descrizione del progetto
    s = add_content_slide(prs, "Descrizione del progetto")
    add_bullets(s, [
        ("Task: deblur + denoise, formulato come problema inverso y = Ax + n", 0),
        ("A: blur gaussiano (sigma=2, kernel 9x9); n: rumore gaussiano additivo, 4 livelli (0.005, 0.01, 0.05, 0.1)", 1),
        ("Dataset: KaoKore v1.3, ritratti RGB 256x256 (9257 immagini)", 0),
        ("Split ufficiale train/dev/test fornito nel labels.csv del dataset: 7405 / 926 / 926", 1),
        ("Obiettivo: implementare e confrontare criticamente metodi di famiglie diverse", 0),
        ("sotto le stesse condizioni sperimentali (stessi input degradati per ogni metodo)", 1),
        ("3 dei 4 metodi proposti (opzione consentita per progetti da uno studente):", 0),
        ("Variazionale (FISTA + Wavelet), Ibrido (PD-Net + TV), End-to-end (UNet)", 1),
    ])

    # 3. Metodologia - overview
    s = add_content_slide(prs, "Metodologia — panoramica")
    add_bullets(s, [
        ("Il problema inverso è mal posto: A attenua le alte frequenze spaziali", 0),
        ("l'inversione ingenua amplifica il rumore -> serve regolarizzazione / un prior", 1),
        ("Tre modi diversi di introdurre questo prior:", 0),
        ("Variazionale: un prior esplicito e scelto a mano (sparsità wavelet), nessun training", 1),
        ("Ibrido: fisica nota (A, gradiente dell'immagine) + poche componenti apprese", 1),
        ("End-to-end: l'intera mappa y -> x è appresa dai dati", 1),
        ("Tutti e tre condividono lo stesso operatore di degradazione e leggono", 0),
        ("esattamente gli stessi file PNG degradati in valutazione (confronto equo)", 1),
    ])

    # 3b. Metodologia: il filo conduttore
    s = add_content_slide(prs, "Metodologia — il filo conduttore")
    add_prose(s, [
        "Tutti e tre i metodi rispondono alla stessa domanda di fondo: dato che A distrugge "
        "informazione (attenua le alte frequenze), come iniettiamo abbastanza conoscenza a "
        "priori su cosa sia un'immagine plausibile per colmare quel vuoto, senza amplificare "
        "il rumore al posto del segnale?",

        "Quello che cambia tra i tre metodi è quanto di quel prior sia scelto a mano oppure "
        "appreso dai dati. FISTA sta a un estremo: il prior è interamente costruito a mano "
        "(le immagini naturali sono sparse in base wavelet), e l'unica cosa tarata è uno "
        "scalare per livello di rumore. UNet sta all'estremo opposto: nessuna informazione "
        "sulla fisica è data alla rete — tutto, inclusa una nozione implicita di come sia un "
        "ritratto KaoKore degradato, viene dedotto solo dagli esempi di training.",

        "PD-Net sta deliberatamente nel mezzo: la fisica già nota esattamente (operatore A, "
        "gradiente dell'immagine) resta fissa dentro la rete, e solo i passi senza forma "
        "chiusa (gli aggiornamenti prossimali di Chambolle-Pock) sono appresi. Per questo "
        "raggiunge l'accuratezza di UNet con due ordini di grandezza in meno di parametri — "
        "non deve mai imparare ciò che la fisica gli dà già gratis.",
    ], size=17)

    # 4. Metodologia: FISTA
    s = add_content_slide(prs, "Metodologia — Variazionale: FISTA + Wavelet")
    add_image(s, "results/figures/formula_fista.png", 1.9, 1.3, height=0.75)
    add_bullets(s, [
        ("W: trasformata wavelet ortogonale", 0),
        ("Convesso ma non liscio -> metodo proximal gradient (Beck & Teboulle, 2009)", 0),
        ("Per W ortogonale, il prossimale del termine L1 = soft-thresholding dei coefficienti wavelet", 0),
        ("Estrapolazione alla Nesterov: velocità di convergenza O(1/k^2) invece di O(1/k)", 0),
        ("Passo 1/L: L = ||A^T A|| = 1 esattamente, grazie alle condizioni al contorno circolari", 0),
        ("(A diventa un operatore circolante simmetrico, diagonalizzato dalla trasformata di Fourier)", 1),
    ], top=2.5)

    # 5. Metodologia: PD-Net
    s = add_content_slide(prs, "Metodologia — Ibrido: PD-Net + TV")
    add_image(s, "results/figures/formula_pdnet.png", 2.4, 1.3, height=0.75)
    add_bullets(s, [
        ("Unrolling dell'algoritmo primal-dual di Chambolle-Pock", 0),
        ("I passi prossimali/di proiezione esatti sono sostituiti da piccole CNN apprese", 0),
        ("La variabile duale vive nello spazio del gradiente (non dei dati): lega la rete", 0),
        ("esplicitamente alla struttura TV, a differenza di un primal-dual generico appreso", 1),
        ("Il gradiente del data-fidelity A^T(Ax-y) resta esatto (non appreso) ad ogni iterazione", 0),
        ("-> ibrido: fisica nota + correzione appresa, non una scatola nera", 1),
    ], top=2.5)

    # 6. Metodologia: UNet
    s = add_content_slide(prs, "Metodologia — End-to-end: UNet")
    add_bullets(s, [
        ("Completamente supervisionato: impara la mappa diretta y -> x dalle coppie di training", 0),
        ("Scelta rispetto a ViT / NAF-Net: la degradazione è spazialmente locale (kernel piccolo,", 0),
        ("rumore per-pixel) -> non serve attention a lungo raggio; più semplice da implementare", 1),
        ("bene nel tempo disponibile", 1),
        ("Encoder-decoder con skip connection: preservano il dettaglio spaziale fine", 0),
        ("attraverso il collo di bottiglia, necessario perché input e output condividono la struttura", 1),
        ("Apprendimento residuale (la rete impara solo la correzione):", 0),
    ])
    add_image(s, "results/figures/formula_unet_residual.png", 2.9, 4.9, height=0.65)

    # 7. Implementazione: FISTA
    s = add_content_slide(prs, "Implementazione — FISTA + Wavelet")
    add_bullets(s, [
        ("Wavelet: Daubechies db4, 3 livelli di decomposizione, mode='periodization'", 0),
        ("(periodization mantiene la DWT esattamente ortogonale, necessario perché il prossimale sia esatto)", 1),
        ("100 iterazioni per immagine, nessun parametro appreso", 0),
        ("lambda tarato con grid search sul dev set (20 immagini), separatamente per livello di rumore", 0),
        ("(0.005 -> 0.0025, 0.01 -> 0.0035, 0.05 -> 0.0175, 0.1 -> 0.07: cresce con il rumore,", 1),
        ("come atteso dal principio di discrepanza)", 1),
    ])

    # 8. Implementazione: PD-Net
    s = add_content_slide(prs, "Implementazione — PD-Net")
    add_image(s, "results/figures/pdnet_diagram.png", 1.3, 1.3, width=10.7)
    add_bullets(s, [
        ("8 iterazioni di unrolling, pesi indipendenti per iterazione; blocchi CNN: Conv3x3-LeakyReLU-Conv3x3", 0),
        ("4 modelli specializzati (uno per livello di rumore), 700 immagini di training, 12 epoche, Adam (lr=2e-4), loss L1", 0),
    ], top=5.9, size=15)

    # 9. Implementazione: UNet
    s = add_content_slide(prs, "Implementazione — UNet")
    add_image(s, "results/figures/unet_diagram.png", 1.9, 1.3, width=9.6)
    add_bullets(s, [
        ("4 livelli di downsampling, canali base 48, GroupNorm (stabile con batch size piccoli)", 0),
        ("4 modelli specializzati (uno per livello di rumore), 1000 immagini di training, 20 epoche, Adam (lr=2e-4), loss L1", 0),
    ], top=5.9, size=15)

    # 10. Introduzione agli esperimenti
    s = add_content_slide(prs, "Introduzione agli esperimenti")
    add_bullets(s, [
        ("Degradazione: blur gaussiano sigma=2, kernel 9x9, condizioni al contorno circolari", 0),
        ("Rumore: gaussiano additivo, std = livello di rumore direttamente (0.005/0.01/0.05/0.1), su immagini in [0,1]", 0),
        ("Osservazioni degradate generate una volta, salvate come PNG a 8 bit -> input identici per tutti i metodi", 0),
        ("Scelte di scope dettate dal tempo (dichiarate esplicitamente, non nascoste):", 0),
        ("valutazione su un sottoinsieme fisso (80 test + 20 dev) invece delle 926/926 totali", 1),
        ("3 metodi su 4 (consentito per progetti da uno studente); scartato il metodo generativo (Diffusion+DPS)", 1),
        ("4 modelli specializzati per metodo appreso, per un confronto equo con il lambda per livello di FISTA", 1),
    ])

    # 11. Risultati - tabella
    s = add_content_slide(prs, "Risultati numerici — tabella riassuntiva")
    rows_data = read_comparison_csv()
    table_rows = [[r["method"], r["noise_level"], f"{float(r['psnr_mean']):.2f}",
                   f"{float(r['ssim_mean']):.3f}", f"{float(r['avg_time_sec']):.4f}"]
                  for r in rows_data]
    add_table(s, ["Metodo", "Livello di rumore", "PSNR (dB)", "SSIM", "Tempo/immagine (s)"], table_rows,
              top=1.3, height=5.7)

    # 12. Risultati - plot
    s = add_content_slide(prs, "Risultati numerici — plot comparativo")
    add_image(s, "results/figures/comparison_plot.png", 1.0, 1.5, width=11.3)

    # 13. Risultati - confronto visivo
    s = add_content_slide(prs, "Risultati numerici — confronto visivo")
    add_image(s, "results/figures/composite_comparison.png", 3.1, 1.2, height=6.15)

    # 13b. Come leggere l'anomalia non monotona
    s = add_content_slide(prs, "Come leggere l'anomalia non monotona")
    add_prose(s, [
        "Un risultato merita uno sguardo più attento invece di essere riportato e basta: sia "
        "UNet che PD-Net raggiungono un PSNR leggermente più basso a sigma=0.005 che a "
        "sigma=0.01 — il caso più facile, meno rumoroso, va peggio. Vale la pena chiedersi "
        "perché, non solo constatarlo.",

        "Entrambi i metodi usano apprendimento residuale: l'output è y più una correzione "
        "appresa. A rumore molto basso quella correzione è minuscola, quindi il segnale di "
        "training che guida l'ottimizzazione è proporzionalmente più debole che a sigma=0.01, "
        "dove la correzione è più grande e più facile da imparare nello stesso numero fisso "
        "di epoche.",

        "Lo stesso pattern compare indipendentemente in due architetture diverse, allenate "
        "separatamente — questo è ciò che lo rende un risultato genuino e non rumore di un "
        "singolo run: indica una proprietà del setup di training, non una coincidenza "
        "dell'inizializzazione. Più epoche al livello di rumore più basso probabilmente "
        "chiuderebbero questo divario.",
    ], size=18)

    # 13c. Cosa ci dice l'efficienza di PD-Net
    s = add_content_slide(prs, "Cosa ci dice l'efficienza di PD-Net")
    add_prose(s, [
        "PD-Net eguaglia l'accuratezza di UNet ad ogni livello di rumore con ~100 volte meno "
        "parametri (302 KB contro 33 MB). Questo è l'argomento empirico centrale a favore "
        "delle architetture ibride fisicamente informate, non un dettaglio implementativo "
        "minore.",

        "Una rete puramente data-driven deve scoprire, solo dagli esempi, che disfare una "
        "convoluzione nota fa parte del compito — parte dei suoi parametri e dei dati di "
        "training sono spesi a ri-derivare qualcosa che era già noto prima ancora di "
        "iniziare il training.",

        "PD-Net riceve A e il gradiente dell'immagine come blocchi fissi ed esatti; le sue "
        "componenti apprese riempiono solo la parte senza forma chiusa. In termini di machine "
        "learning: un forte inductive bias sostituisce dati e parametri — e ci si aspetta "
        "generalizzi meglio fuori dalle condizioni esatte di training, perché parte del suo "
        "comportamento è garantito, non appreso.",
    ], size=18)

    # 13d. Sull'onestà nel dichiarare i limiti
    s = add_content_slide(prs, "Sull'onestà nel dichiarare i nostri limiti")
    add_prose(s, [
        "Questo progetto è stato svolto con un vincolo di tempo reale e fisso, e diverse "
        "scelte di scope sono state fatte esplicitamente per renderlo possibile: un "
        "sottoinsieme test di 80 immagini invece di 926, tre dei quattro metodi proposti, "
        "un training set e un numero di epoche ridotti.",

        "Niente di questo è nascosto dentro i numeri — è dichiarato apertamente nel report "
        "e nella documentazione del codice, perché un risultato con limiti noti è più "
        "affidabile di uno che sembra completo ma nasconde scorciatoie.",

        "Allenare silenziosamente su meno immagini senza dirlo avrebbe prodotto numeri più "
        "lucidi in apparenza, ma non avrebbe retto a una domanda diretta sulla metodologia. "
        "Dichiarare i limiti è ciò che rende ogni numero di questa presentazione difendibile.",
    ], size=18)

    # 14. Conclusioni
    s = add_content_slide(prs, "Conclusioni")
    add_bullets(s, [
        ("UNet e PD-Net superano FISTA di 2-3 dB di PSNR ad ogni livello di rumore:", 0),
        ("un prior appreso sulla statistica di KaoKore batte un prior generico di sparsità wavelet", 1),
        ("UNet e PD-Net ottengono accuratezza quasi identica, ma PD-Net usa ~100 volte meno parametri", 0),
        ("(checkpoint 302 KB vs 33 MB): incorporare la fisica nota riduce cosa la rete deve imparare", 1),
        ("FISTA è ~40 volte più lento per immagine (ottimizzazione iterativa vs un solo forward pass),", 0),
        ("ma non richiede training ed è completamente interpretabile, a differenza dei due metodi appresi", 1),
        ("Limiti: sottoinsieme di valutazione e training set/epoche ridotti, per vincoli di tempo", 0),
        ("Sviluppi futuri: test set e training set completi, più epoche, aggiungere il 4° metodo", 0),
        ("(Diffusion + DPS), provare NAF-Net come architettura end-to-end alternativa", 1),
    ])

    prs.save("presentation/presentazione_ita.pptx")
    print("saved presentation/presentazione_ita.pptx")


if __name__ == "__main__":
    build()
