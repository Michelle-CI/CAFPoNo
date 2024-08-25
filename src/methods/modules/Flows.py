
from typing import Any, List, Tuple
import torch
import torch.nn as nn 
from math import pi 

# Constant value
EPS = 1e-6
BASE_DIST_DICT = {
    'normal': torch.distributions.normal.Normal(0, 1.0),
    'laplace': torch.distributions.Laplace(0, 1.0), 
    'gumbel': torch.distributions.Gumbel(0, 1.0) 
}

def MLP(d_in: int, d_out: int, hidden_sizes: List[int]=None):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [d_in] + (hidden_sizes or []) + [d_out]

    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout1d(p=0.1))
            # layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
    
    return nn.Sequential(*layers)

class AffineFlow(nn.Module):
    def __init__(self, base_dist='normal') -> None:
        super(AffineFlow, self).__init__()
        self.base_dist = BASE_DIST_DICT[base_dist]
        self.h_net = MLP(d_in=1, d_out=1, hidden_sizes=[12]*3)
        self.s = nn.Parameter(data=torch.randn(())) 
        self.t = nn.Parameter(data=torch.randn(())) 

    def forward(self, X, Y):
        """
        X: nx1
        Y: nx1
        """
        h = self.h_net(X)
        noise = (Y - self.t) / torch.exp(self.s) - h
        log_pe = self.base_dist.log_prob(noise)
        log_det = -self.s

        return log_pe, log_det

class SingleCDFFlow(nn.Module):
    """
    Model P(x) through a normal distribution noise. 
    f(x) = noise, noise ~ N(0, 1)
    f(x) = sigmoid_inverse(t) 
    t = weight sum of CDF(X, mu, sigma)
    """
    def __init__(self, k=1, base_dist='normal', **kwargs) -> None:
        super(SingleCDFFlow, self).__init__()
        self.k = k
        self.w_logit = nn.Parameter(data=torch.randn(self.k))
        self.mu = nn.Parameter(data=torch.randn(self.k))
        self.log_sigma = nn.Parameter(data=torch.randn(self.k))
        self.softmax = nn.Softmax(dim=0)

    def flow(self, X):
        w = self.softmax(self.w_logit) # (k)
        sigma = torch.exp(self.log_sigma) # (k)
        dist = torch.distributions.normal.Normal(self.mu, sigma)
        cdf = dist.cdf(X) # (N, k)
        z = torch.clip((w * cdf).sum(axis=1, keepdims=True), EPS, 1 - EPS) # (N)
        return w, dist, z

    def forward(self, X):
        w, dist, z = self.flow(X)
        dist_pdf = torch.clip(torch.exp(dist.log_prob(X)), 1e-12, None)
        log_det = torch.log((w * dist_pdf).sum(axis=1, keepdims=True))
        log_px = log_det
        return log_px, 0, log_det
        

class CDFFlow(nn.Module):
    def __init__(self, hidden_sizes, nx=1, k=1, base_dist='normal', **kwargs) -> None:
        super(CDFFlow, self).__init__()
        self.base_dist = BASE_DIST_DICT[base_dist]
        self.k = k
        self.h_net = MLP(d_in=nx, d_out=1, hidden_sizes=hidden_sizes)
        self.g_inverse_w_logit = nn.Parameter(data=torch.randn(self.k))     # (k)
        self.g_inverse_mu = nn.Parameter(data=torch.randn(self.k))          # (k)
        self.g_inverse_log_sigma = nn.Parameter(data=torch.randn(self.k))   # (k)
        self.softmax = nn.Softmax(dim=0)


    def g_inverse(self, Y): # Y: (N, 1)
        w = self.softmax(self.g_inverse_w_logit) # (k)
        sigma = torch.exp(self.g_inverse_log_sigma) # (k)
        sigma = torch.clip(sigma, 1e-6, None)
        dist = torch.distributions.normal.Normal(self.g_inverse_mu, sigma)
        cdf = dist.cdf(Y) # (N, k)
        t = torch.clip((w * cdf).sum(axis=1, keepdims=True), EPS, 1 - EPS) # (N, 1)
        sigmoid_inverse = -torch.log(1/t - 1) # (N, 1)
        return w, dist, t, sigmoid_inverse

    def forward(self, X, Y):
        """
        X: nx1
        Y: nx1
        """
        w, dist, t, sigmoid_inverse = self.g_inverse(Y)
        noise = sigmoid_inverse - self.h_net(X)
        # print(f'{t = } \n{sigmoid_inverse = } \n{noise = }')

        log_pe = self.base_dist.log_prob(noise)
        dist_pdf = torch.exp(torch.clip(dist.log_prob(Y), -12, 12))
        # dist_pdf = torch.clip(torch.exp(dist.log_prob(Y)), 1e-12, None)
        dt_dy = torch.abs((w * dist_pdf).sum(axis=1, keepdims=True))
        dgi_dt = torch.abs(1/(t * (1-t)))
        log_det = torch.log(dgi_dt) + torch.log(dt_dy)
        log_pyx = log_pe + log_det
        return log_pyx, noise, log_pe, log_det 

if __name__ == '__main__':
    X = torch.randn((128, 1))
    Y = torch.randn((128, 1))
    model = CDFFlow(k=3)
    print(model(X, Y))