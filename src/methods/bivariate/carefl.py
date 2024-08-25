# Causal Autogressive Flow

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform
from torch.utils.data import DataLoader, Dataset

from src.methods.modules.nets import MLP1layer
from src.methods.modules.nflib import AffineCL, NormalizingFlowModel

class CustomSyntheticDatasetDensity(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]

    # print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.data_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index]

    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }

class CAREFLModel:
    def __init__(self, n_layers, n_hidden, epochs, verbose, batch_size, device='cuda') -> None:
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size

        self.device = device
        self.dim = None
        self.direction = 'none'
        self.flow_xy = self.flow_yx = self.flow = None
        self._nhxy = self._nhyx = self._nlxy = self._nlyx = None

    def flow_lr(self, data):
        dset, test_dset, dim = self._get_datasets(data)
        self.dim = dim
        # Conditional Flow Model: X->Y
        flows_xy, _ = self._train(dset)
        self.flow_xy, score_xy, self._nlxy, self._nhxy = self._evaluate(flows_xy, test_dset)
        # Conditional Flow Model: Y->X
        flows_yx, _ = self._train(dset, parity=True)
        self.flow_yx, score_yx, self._nlyx, self._nhyx = self._evaluate(flows_yx, test_dset, parity=True)
        return score_xy, score_yx

    def _get_datasets(self, input, training_split=0.8):
        """
        Check data type, which can be:
            - an np.ndarray, in which case split it and wrap it into a train Dataset and and a test Dataset
            - a Dataset, in which case duplicate it (test dataset is the same as train dataset)
            - a tuple of Datasets, in which case just return.
        return a train Dataset, and a test Dataset
        """
        assert isinstance(input, (np.ndarray, Dataset, tuple, list))
        # training_split = 0.8
        dim = input.shape[-1]
        if training_split == 1.:
            data_test = np.copy(input)
        else:
            data_test = np.copy(input[int(training_split * input.shape[0]):])
            input = input[:int(training_split * input.shape[0])]
        dset = CustomSyntheticDatasetDensity(input.astype(np.float32))
        test_dset = CustomSyntheticDatasetDensity(data_test.astype(np.float32))
        return dset, test_dset, dim
       

    def _train(self, dset, parity=False):
        """
        Train one or multiple flors for a single direction, specified by `parity`.
        """
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.batch_size)
        flows = self._get_flow_arch(parity)
        all_loss_vals = []
        for flow in flows:
            optimizer, scheduler = self._get_optimizer(flow.parameters())
            flow.train()
            loss_vals = []
            for e in range(self.epochs):
                loss_val = 0
                for _, x in enumerate(train_loader):
                    x = x.to(self.device)
                    # compute loss
                    _, prior_logprob, log_det = flow(x)
                    loss = - torch.sum(prior_logprob + log_det)
                    loss_val += loss.item()
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss_val / len(train_loader))
                if self.verbose:
                    print('epoch {}/{} \tloss: {}'.format(e, self.epochs, loss_val))
                loss_vals.append(loss_val)
            all_loss_vals.append(loss_vals)
        return flows, all_loss_vals

    def _get_flow_arch(self, parity=False):
        """
        Returns a normalizing flow according to the config file.

        Parameters:
        ----------
        parity: bool
            If True, the flow follows the (1, 2) permutations, otherwise it follows the (2, 1) permutation.
        """
        # this method only gets called by _train, which in turn is only called after self.dim has been initialized
        dim = self.dim
        # Base dist
        prior = Laplace(torch.zeros(dim).to(self.device), torch.ones(dim).to(self.device))
        net_class = MLP1layer

        # flow type
        def ar_flow(hidden_dim):
            return AffineCL(dim=dim, nh=hidden_dim, scale_base=True,
                                shift_base=True, net_class=net_class, parity=parity,
                                scale=True)

        # support training multiple flows for varying depth and width, and keep only best
        self.n_layers = self.n_layers if type(self.n_layers) is list else [self.n_layers]
        self.n_hidden = self.n_hidden if type(self.n_hidden) is list else [self.n_hidden]
        normalizing_flows = []
        for nl in self.n_layers:
            for nh in self.n_hidden:
                # construct normalizing flows
                flow_list = [ar_flow(nh) for _ in range(nl)]
                normalizing_flows.append(NormalizingFlowModel(prior, flow_list).to(self.device))
        return normalizing_flows

    def _get_optimizer(self, parameters):
        """
        Returns an optimizer according to the config file
        """
        optimizer = optim.Adam(parameters, lr=1e-3, weight_decay=0.00, betas=(0.9, 0.999), amsgrad=False)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=self.verbose)
        return optimizer, scheduler

    def _get_params_from_idx(self, idx):
        return self.n_layers[idx // len(self.n_hidden)], self.n_hidden[idx % len(self.n_hidden)]

    def _evaluate(self, flows, test_dset, parity=False):
        """
        Evaluate a set of flows on test dataset, and return the one with best test likelihood.
        """
        loader = DataLoader(test_dset, batch_size=128)
        scores = []
        for idx, flow in enumerate(flows):
            score = np.nanmean(np.concatenate([flow.log_likelihood(x.to(self.device)) for x in loader]))
            scores.append(score)
        try:
            # in case all scores are nan, this will raise a ValueError
            idx = np.nanargmax(scores)
        except ValueError:
            # arbitrarily pick flows[0], this doesn't matter since best_score = nan, which will
            idx = 0
        # unlike nanargmax, nanmax only raises a RuntimeWarning when all scores are nan, and will return nan
        best_score = np.nanmax(scores)
        best_flow = flows[idx]
        nl, nh = self._get_params_from_idx(idx)  # for debug
        return best_flow, best_score, nl, nh

def CAREFL(X, Y, args):
    with torch.random.fork_rng():
        torch.random.manual_seed(args.seed)
        X, Y = map(lambda x: (x - x.mean())/x.std(), (X, Y)) 
        model = CAREFLModel(n_layers=args.nl, n_hidden=args.nh, epochs=args.epochs, verbose=args.verbose, batch_size=args.batch_size, device=args.device)
        data = np.stack([X, Y], axis=1)
        score_XY, score_YX = model.flow_lr(data)
        return score_XY, score_YX