import torch
from torch import nn, Tensor
from typing import Any, TypeVar, cast, Callable, Sequence, Iterable

def gram_Gaussian(x: Tensor, sigma: float = 1.) -> Tensor:
    """Gram matrix with Gaussian kernel.

    Parameters
    ----------
    x : Tensor
        Samples.
    sigma : float, optional
        Bandwidth of the Gaussian kernel, by default 1.

    Returns
    -------
    Tensor
        Gram matrix.
    """
    x = x[:, None]
    d = torch.abs(x - x.t())
    return torch.exp(-d**2 / sigma**2)


def standardize(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Standardize samples over the first dimension.

    Parameters
    ----------
    x : Tensor
        Samples. x.shape = (#samples, #variables)
    eps : float, optional
        epsilon, by default 1e-5

    Returns
    -------
    Tensor
        Standardized samples.
    """
    return (x - x.mean(0)) / (x.std(0, unbiased=False) + eps)

def HSIC(x1: Tensor, x2: Tensor) -> Tensor:
    n = len(x1)
    g1 = gram_Gaussian(x1.t()[0])
    g2 = gram_Gaussian(x2.t()[0])
    h = torch.eye(n, device=x1.device) - 1./n
    ehsic = 1./(n**2) * torch.sum(torch.mm(torch.mm(h, g1), h).t() * g2)
    return ehsic

def max_Gaussian_eHSIC(x1s: Tensor, x2s: Tensor) -> Tensor:
    """Maximum empirical HSIC.

    Parameters
    ----------
    x1s : Tensor
        Samples. x1.shape = (#samples, #variables).
    x2s : Tensor
        Samples. x2.shape = (#samples, #variables).

    Returns
    -------
    Tensor
        _description_
    """
    n = len(x1s)
    ehsic = None

    for x1 in x1s.t():
        g1 = gram_Gaussian(x1)
        for x2 in x2s.t():
            g2 = gram_Gaussian(x2)  
            h = torch.eye(n, device=x1.device) - 1./n
            ehsic_new = 1./(n**2) * \
                torch.sum(torch.mm(torch.mm(h, g1), h).t() * g2)

            if (ehsic is None) or (ehsic_new > ehsic):
                ehsic = ehsic_new
    return cast(Tensor, ehsic)