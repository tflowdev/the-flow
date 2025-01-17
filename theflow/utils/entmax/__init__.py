__version__ = "1.1.dev0"

from theflow.utils.entmax.activations import entmax15, Entmax15, sparsemax, Sparsemax
from theflow.utils.entmax.losses import (
    entmax15_loss,
    Entmax15Loss,
    entmax_bisect_loss,
    EntmaxBisectLoss,
    sparsemax_bisect_loss,
    sparsemax_loss,
    SparsemaxBisectLoss,
    SparsemaxLoss,
)
from theflow.utils.entmax.root_finding import entmax_bisect, EntmaxBisect, sparsemax_bisect, SparsemaxBisect

__all__ = [
    "entmax15",
    "Entmax15",
    "sparsemax",
    "Sparsemax",
    "entmax15_loss",
    "Entmax15Loss",
    "entmax_bisect_loss",
    "EntmaxBisectLoss",
    "sparsemax_bisect_loss",
    "sparsemax_loss",
    "SparsemaxBisectLoss",
    "SparsemaxLoss",
    "entmax_bisect",
    "EntmaxBisect",
    "sparsemax_bisect",
    "SparsemaxBisect",
]
