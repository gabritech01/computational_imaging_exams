"""FISTA (Beck & Teboulle, 2009) per il problema di deblur regolarizzato con sparsità
wavelet: min_x 1/2||Ax-y||^2 + lambda*||Wx||_1, con W trasformata wavelet ortogonale.
Poiché W è ortogonale, l'operatore prossimale di lambda*||Wx||_1 si riduce al
soft-thresholding diretto dei coefficienti wavelet (vedi NOTE_ORALE.md)."""
import numpy as np
import pywt
import torch

from src.degradation.operators import Blurring


def _threshold_channel(img: np.ndarray, lam: float, wavelet: str, level: int) -> np.ndarray:
    # Uso mode="periodization" invece del default di pywt (symmetric): è l'unica
    # modalità di bordo che mantiene la DWT esattamente ortogonale, il che rende il
    # soft-thresholding il vero operatore prossimale qui sotto, non solo un'euristica.
    coeffs = pywt.wavedec2(img, wavelet, mode="periodization", level=level)
    approx, details = coeffs[0], coeffs[1:]
    # Riduco solo i coefficienti di dettaglio e lascio intatta l'approssimazione
    # grossolana, perché è lì che sta il contenuto a bassa frequenza dell'immagine.
    shrunk = [approx] + [
        tuple(pywt.threshold(c, lam, mode="soft") for c in triplet) for triplet in details
    ]
    return pywt.waverec2(shrunk, wavelet, mode="periodization")


def wavelet_soft_threshold(x: torch.Tensor, lam: float, wavelet: str = "db4", level: int = 3) -> torch.Tensor:
    """Applica una DWT 2D per canale, riduce solo i coefficienti di dettaglio
    (l'approssimazione grossolana resta intatta), e ricostruisce. x: (C,H,W)."""
    arr = x.detach().cpu().numpy()
    out = np.stack([_threshold_channel(arr[c], lam, wavelet, level) for c in range(arr.shape[0])])
    return torch.from_numpy(out).to(x.dtype)


def fista_deblur(y: torch.Tensor, blur: Blurring, lam: float, n_iter: int = 100,
                  wavelet: str = "db4", level: int = 3) -> torch.Tensor:
    """Risolve il problema per una singola osservazione y (C,H,W). Il passo è 1/L
    con L=1: sotto condizioni al contorno circolari il blur gaussiano è un operatore
    circolante simmetrico, il cui massimo valore singolare è esattamente 1 (il kernel
    somma a 1, con il massimo raggiunto alla frequenza zero), quindi non serve nessuna
    power iteration per stimare L."""
    # Inizializzo sia x che il punto estrapolato z con l'osservazione stessa: un punto
    # di partenza economico e ragionevole, dato che y è già vicina a x.
    x_prev = y.clone()
    z = y.clone()
    t_prev = 1.0
    for _ in range(n_iter):
        # passo di gradiente sul termine liscio di data-fidelity, passo 1/L con L=1
        # (vedi docstring sopra) -- non serve nessuna line search perché L è esatto
        grad = blur._adjoint(blur._matvec(z.unsqueeze(0)) - y.unsqueeze(0))[0]
        x = wavelet_soft_threshold(z - grad, lam, wavelet, level)
        # aggiornamento del momentum alla Nesterov: è l'unica differenza rispetto a
        # ISTA, ed è quello che porta la convergenza da O(1/k) a O(1/k^2)
        t = (1 + (1 + 4 * t_prev**2) ** 0.5) / 2
        z = x + ((t_prev - 1) / t) * (x - x_prev)
        x_prev, t_prev = x, t
    # Applico il clamp solo alla fine, non durante le iterazioni, così l'ottimizzazione
    # non viene distorta da un vincolo che il modello matematico non prevede
    return x_prev.clamp(0, 1)
