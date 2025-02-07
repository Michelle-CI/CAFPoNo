import os
import sys

from tqdm import trange
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy import stats
from torch.utils.data import Dataset, DataLoader

from src.utils import standardize
from src.HSIC import HSIC
from src.methods.modules.Flows import MLP

class PairDataset(Dataset):

    def __init__(self, data):
        super(PairDataset, self).__init__()
        self.data = data
        self.num_data = data.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        return self.data[index, :]


# class MLP(nn.Module):
#     """
#     Python implementation MLP, which is the same of G1 and G2
#     Input: X (x1 or x2)
#     """

#     def __init__(self, n_inputs, n_outputs, n_layers=1, n_units=100):
#         """ The MLP must have the first and last layers as FC.
#         :param n_inputs: input dim
#         :param n_outputs: output dim
#         :param n_layers: layer num = n_layers + 2
#         :param n_units: the dimension of hidden layers
#         :param nonlinear: nonlinear function
#         """
#         super(MLP, self).__init__()
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.n_layers = n_layers
#         self.n_units = n_units

#         # create layers
#         layers = [nn.Linear(n_inputs, n_units)]
#         for _ in range(n_layers):
#             layers.append(nn.ReLU())
#             layers.append(nn.Linear(n_units, n_units))
#         layers.append(nn.ReLU())
#         layers.append(nn.Linear(n_units, n_outputs))
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layers(x)
#         return x

class PNL_model(object):
    """
    Use of constrained nonlinear ICA for distinguishing cause from effect.
    Python Version 3.7
    PURPOSE:
          To find which one of xi (i=1,2) is the cause. In particular, this
          function does
            1) preprocessing to make xi rather close to Gaussian,
            2) learn the corresponding 'disturbance' under each assumed causal
            direction, and
            3) performs the independence tests to see if the assumed cause if
            independent from the learned disturbance.
    """

    def __init__(self, batch_size, hidden_sizes, epochs=3000, verbose=False, **kwargs):
        '''
        Construct the PNL model.

        Parameters:
        ----------
        epochs: training epochs.
        '''

        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_sizes = hidden_sizes
        self.verbose = verbose

    def nica_mnd(self, X, TotalEpoch,):
        """
        Use of "Nonlinear ICA" for distinguishing cause from effect
        PURPOSE: Performing nonlinear ICA.

        Parameters
        ----------
        X (n*T): a matrix containing multivariate observed data. Each row of the matrix X is a observed signal.

        Returns
        ---------
        Y (n*T): the separation result.
        """
        X = X.astype(np.float32)

        train_dataset = PairDataset(X)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True)

        G1 = MLP(1, 1, hidden_sizes=self.hidden_sizes)
        G2 = MLP(1, 1, hidden_sizes=self.hidden_sizes)
        optimizer = torch.optim.Adam([
            {'params': G1.parameters()},
            {'params': G2.parameters()}], lr=1e-4, betas=(0.9, 0.99))

        with trange(TotalEpoch, desc='Training', disable=not self.verbose) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                for x_batch in train_loader:

                    x1, x2 = x_batch[:,0].reshape(-1,1), x_batch[:,1].reshape(-1,1)
                    x1.requires_grad = True
                    x2.requires_grad = True
                    
                    e = G2(x2) - G1(x1)
                    loss_pdf = 0.5 * torch.sum(e**2)

                    jacob = autograd.grad(outputs=e, inputs=x2, grad_outputs=torch.ones(e.shape), create_graph=True,
                                        retain_graph=True, only_inputs=True)[0]
                    loss_jacob = - torch.sum(torch.log(torch.abs(jacob) + 1e-16))

                    loss = loss_jacob + loss_pdf

                    loss.backward()
                    optimizer.step()
        
        X1_all = torch.tensor(X[:, 0].reshape(-1,1))
        X2_all = torch.tensor(X[:, 1].reshape(-1,1))
        e_estimated = G2(X2_all) - G1(X1_all)

        return X1_all, e_estimated

    def cause_or_effect(self, data_x, data_y,):
        '''
        Fit a PNL model in two directions and test the independence between the input and estimated noise

        Parameters
        ---------
        data_x: input data (nx1)
        data_y: output data (nx1)

        Returns
        ---------
        pval_forward: p value in the x->y direction
        pval_backward: p value in the y->x direction
        '''
        torch.manual_seed(0)
        # Now let's see if x1 -> x2 is plausible
        data = np.concatenate((data_x, data_y), axis=1)
        # print('To see if x1 -> x2...')
        y1, y2 = self.nica_mnd(data, self.epochs,)

        y1_np = y1.detach()
        y2_np = y2.detach()

        _, pval_forward = stats.ttest_ind(y1_np, y2_np)
        hsic_forward = HSIC(y1_np, y2_np)

        # Now let's see if x2 -> x1 is plausible
        # print('To see if x2 -> x1...')
        y1, y2 = self.nica_mnd(data[:, [1, 0]], self.epochs,)
        
        y1_np = y1.detach()
        y2_np = y2.detach()

        _, pval_backward = stats.ttest_ind(y1_np, y2_np)
        hsic_backward = HSIC(y1_np, y2_np)
 
        # return pval_forward[0], pval_backward[0]
        return -hsic_forward, -hsic_backward

def PNL(X, Y, args):
    with torch.random.fork_rng():
        torch.random.manual_seed(args.seed)
        model = PNL_model(**vars(args))
        X, Y = map(standardize, (X, Y))
        X, Y = map(lambda x: x.reshape(-1, 1), (X, Y))
        score_XY, score_YX = model.cause_or_effect(X, Y,)
        score_XY, score_YX = map(lambda ts: ts.item(), (score_XY, score_YX))
        return score_XY, score_YX
