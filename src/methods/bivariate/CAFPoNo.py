import sys
import os.path
from typing import Any
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from src.utils import standardize
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.methods.modules.Flows import AffineFlow, CDFFlow, SingleCDFFlow
from scipy import stats
from src.HSIC import HSIC, max_Gaussian_eHSIC
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)



class FlowLightning(LightningModule):
    def __init__(self, model_Y, verbose, weight_decay, nx = 1, model_X=None, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_X:
            self.flow_X = eval(model_X['model_name'])(**model_X)
        self.flow_Y = eval(model_Y['model_name'])(nx = 1, **model_Y)

    def configure_optimizers(self):
        optim = Adam(self.parameters(recurse=True), lr=1e-3, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optim, factor=.1, patience=10, verbose=self.hparams.verbose,)
        return {
            "optimizer": optim,
            # "lr_scheduler": scheduler,
            # "monitor": "train_loss"
        }

    def loss(self, X, Y):
        # log_px, *_ = self.flow_X(X)
        log_py, *_ = self.flow_Y(X, Y)
        loss = -(log_py).mean()
        return loss
    
    def likelihood_score(self, X, Y):
        # self.flow_X.eval()
        self.flow_Y.eval()
        loss = -self.loss(X, Y)
        return loss.item()
    
    def ind_score(self, X, Y):
        self.flow_Y.eval()
        _, estimated_noise, *_ = self.flow_Y(X, Y)
        stat = HSIC(X, estimated_noise).item()
        return -stat
    
    def estimate_noise(self, X, Y):
        self.flow_Y.eval()
        _, estimated_noise, *_ = self.flow_Y(X, Y)
        return estimated_noise.detach().numpy()

    def training_step(self, batch, batch_idx):
        X, Y = batch
        loss = self.loss(X, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        loss = self.loss(X, Y)
        self.log('val_loss', loss, on_epoch=True,prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        X, Y = batch
        # score = self.likelihood_score(X, Y)
        score = self.ind_score(X, Y)
        loss = self.loss(X, Y)
        self.log('test_score', score, on_epoch=True,)
        self.log('test_loss', loss, on_epoch=True,)

    def predict_step(self, batch, batch_idx):
        X, Y = batch
        _, est_noise, *_ = self.flow_Y(X, Y)
        return X, est_noise
    
    def g_inverse(self, Y):
        self.flow_Y.eval()
        *_, z = self.flow_Y.g_inverse(Y)
        return z.detach().numpy()

    def hX(self, X):
        self.flow_Y.eval()
        hx = self.flow_Y.h_net(X)
        return hx.detach().numpy()

    def fit(self, X, Y, batch_size, max_epochs, callbacks=None):
        callbacks = callbacks or []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            dataset = TensorDataset(X, Y)
            train_ratio = 0.5
            val_ratio = 0.1
            test_ratio = 0.4
            train_set, val_set, test_set = random_split(dataset, [train_ratio, val_ratio, test_ratio])
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size)
            test_loader = DataLoader(test_set, batch_size=batch_size)

            early_stopping = EarlyStopping(
                mode='min',
                monitor='val_loss',
                patience=10,
                verbose=True,
            ) 
            callbacks.append(early_stopping)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                max_epochs=max_epochs,
                logger=self.hparams.verbose,
                enable_checkpointing=self.hparams.verbose,
                enable_progress_bar=self.hparams.verbose,
                enable_model_summary=self.hparams.verbose,
                deterministic=True,
                callbacks=callbacks,
                detect_anomaly=False,
                gradient_clip_val=1,
                gradient_clip_algorithm="value",
                check_val_every_n_epoch=1,
            )
            trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader) 
            # test_res = trainer.test(model=self, dataloaders=test_loader, verbose=False)
            # test_score = test_res[0]["test_score"]
            test_res = trainer.predict(model=self, dataloaders=test_loader)
            X_test = torch.concat([x[0] for x in test_res], axis=0)
            self.est_noise = torch.concat([x[1] for x in test_res], axis=0)
            self.X_test = X_test
            test_score = -HSIC(X_test, self.est_noise).item()
        return test_score

def CAFPoNo(X, Y, args):
    with torch.random.fork_rng():
        torch.random.manual_seed(args.seed)
        X, Y = map(standardize, (X, Y))
        X, Y = map(lambda x: x.reshape(-1, 1), (X, Y))
        tensor_X = torch.tensor(X, dtype=torch.float32)
        tensor_Y = torch.tensor(Y, dtype=torch.float32)

        # Fit and get the score for X -> Y
        flow_XY = FlowLightning(model_X=args.model_X, model_Y=args.model_Y, verbose=args.verbose, weight_decay=args.weight_decay,)
        score_XY = flow_XY.fit(tensor_X, tensor_Y, batch_size=args.batch_size, max_epochs=args.epochs)
        # score_XY = flow_XY.likelihood_score(tensor_X, tensor_Y)

        # Fit and get the score for Y -> X
        flow_YX = FlowLightning(model_X=args.model_X, model_Y=args.model_Y, verbose=args.verbose, weight_decay=args.weight_decay,)
        score_YX = flow_YX.fit(tensor_Y, tensor_X, batch_size=args.batch_size, max_epochs=args.epochs)
        # score_YX = flow_YX.likelihood_score(tensor_Y, tensor_X)
        
        return score_XY, score_YX