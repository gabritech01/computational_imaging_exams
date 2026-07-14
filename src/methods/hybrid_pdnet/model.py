"""Learned Primal-Dual (PD-Net) con struttura TV esplicita: srotola l'algoritmo di
Chambolle-Pock per min_x 1/2||Ax-y||^2 + lambda*TV(x), sostituendo gli operatori
prossimali esatti con piccoli blocchi CNN appresi. La variabile duale vive nello
spazio del gradiente (tramite l'operatore Gradient), il che lega questa rete
specificamente alla regolarizzazione TV, invece di un prior generico appreso nello
spazio dei dati."""
import torch
import torch.nn as nn

from src.degradation.operators import Blurring, Gradient


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))


class PDNet(nn.Module):
    def __init__(self, blur: Blurring, channels: int = 3, num_iterations: int = 8, mid_channels: int = 32):
        super().__init__()
        self.blur = blur
        self.grad = Gradient()
        self.num_iterations = num_iterations
        self.channels = channels

        # Il Gradient produce 2 canali (orizzontale+verticale) per ogni canale di
        # input: la variabile duale p vive in questo spazio, non in quello dei dati.
        dual_ch = 2 * channels
        # Do pesi indipendenti ad ogni iterazione (una CNNBlock diversa per k), non
        # condivisi: è quello che rende questo un vero "unrolling" e non un ciclo
        # ricorrente, dando più capacità alla rete di specializzarsi per ogni fase.
        self.dual_nets = nn.ModuleList([
            CNNBlock(dual_ch * 2, dual_ch, mid_channels) for _ in range(num_iterations)
        ])
        self.primal_nets = nn.ModuleList([
            CNNBlock(channels * 3, channels, mid_channels) for _ in range(num_iterations)
        ])

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Inizializzo x con l'aggiunto di A applicato a y (come nel Chambolle-Pock
        # classico), non con y stessa: è una stima già ragionevole di x prima ancora
        # di iniziare a iterare. La variabile duale p parte da zero.
        x = self.blur._adjoint(y)
        x_bar = x
        p = torch.zeros(y.shape[0], 2 * self.channels, y.shape[2], y.shape[3], device=y.device, dtype=y.dtype)

        for k in range(self.num_iterations):
            # aggiornamento duale: al posto della proiezione prossimale esatta della
            # TV, uso una CNN che impara un regolarizzatore più espressivo della TV
            gx_bar = self.grad._matvec(x_bar)
            p = p + self.dual_nets[k](torch.cat([p, gx_bar], dim=1))

            # il gradiente del data-fidelity resta calcolato esattamente (fisica nota,
            # non appresa) -- solo il passo di aggiornamento primale è imparato
            data_grad = self.blur._adjoint(self.blur._matvec(x) - y)
            gtp = self.grad._adjoint(p)
            x_new = x + self.primal_nets[k](torch.cat([x, data_grad, gtp], dim=1))

            # extrapolation, come nel Chambolle-Pock classico (qui con theta=1): è
            # quello che accelera la convergenza rispetto a un semplice update primale
            x_bar = x_new + (x_new - x)
            x = x_new

        return x
