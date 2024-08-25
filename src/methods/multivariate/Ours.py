from torch.multiprocessing import Pool, set_start_method
import multiprocessing as mp
import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm
from src.methods.modules.Flows import AffineFlow, CDFFlow, SingleCDFFlow
from src.HSIC import max_Gaussian_eHSIC
from src.utils import full_dag

class FlowMultiLightning(LightningModule):
    def __init__(self, model_Y, nx, verbose, weight_decay, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.flow_Y = eval(model_Y['model_name'])(nx=nx, **model_Y)

    def configure_optimizers(self):
        optim = Adam(self.parameters(recurse=True), lr=1e-3, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optim,}

    def loss(self, X, Y):
        log_py, *_ = self.flow_Y(X, Y)
        loss = -(log_py).mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        X, Y = batch
        loss = self.loss(X, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        loss = self.loss(X, Y)
        self.log('val_loss', loss, on_epoch=True,prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        X, Y = batch
        _, est_noise, *_ = self.flow_Y(X, Y)
        return est_noise

    def transform(self, Y):
        *_, g_inverse = self.flow_Y.g_inverse(Y)
        return g_inverse.detach().numpy()

    def fit(self, data, batch_size, max_epochs, callbacks=None):
        callbacks = callbacks or []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # X.shape = (#samples, #variables), Y.shape = (#samples, 1)
            X_train, Y_train, X_test, Y_test = data
            train_set = TensorDataset(X_train, Y_train)
            test_set = TensorDataset(X_test, Y_test)
            val_ratio = 0.2
            train_set, val_set = random_split(train_set, [1 - val_ratio, val_ratio])
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
            test_res = trainer.predict(model=self, dataloaders=test_loader)
            est_noise = torch.concat(test_res, axis=0)
            score = -max_Gaussian_eHSIC(X_test, est_noise).item()
            return score

class OursTrainer:
    def __init__(self, args, **kwargs) -> None:
        self.args = args

    def fit(self, data, prunning=False):
        self.data = (data - data.mean(axis=0)) / data.std(axis=0)
        train_size = int(self.data.shape[0] * self.args.train_ratio)
        self.train_data = self.data[:train_size]
        self.test_data = self.data[train_size:]
        if len(self.test_data) == 0: 
            self.test_data = self.data

        variables = list(range(self.args.num_variables))
        self.causal_order = []
        for _ in tqdm(range(self.args.num_variables)):
            sink = self.find_sink(variables)
            self.causal_order.append(sink)
            # self.causal_order = [sink] + self.causal_order
            variables = [x for x in variables if x != sink]
        if prunning:
            dag = self.pruning(threshold=0.001)
        else:
            dag = full_dag(self.causal_order)
        return dag, self.causal_order

    def train_one(self, task):
        # Concatenate Y_train and Y_test
        node, nx, data = task
        effect = torch.concatenate((data[1], data[3]), axis=0)
        with torch.random.fork_rng():
            torch.random.manual_seed(self.args.seed) 
            model = FlowMultiLightning(model_Y = self.args.model_Y, nx=nx, verbose=self.args.verbose, weight_decay=self.args.weight_decay)
            ind_score = model.fit(data=data,
                            batch_size=self.args.batch_size,
                            max_epochs=self.args.epochs)
            g_inverse = model.transform(effect)
        return (node, ind_score, g_inverse)

    def train_parallel(self, variables):
        tasks = []
        for y in variables:
            X = [x for x in variables if x != y] 
            X_train, X_test = map(lambda data: data[:, X], (self.train_data, self.test_data))
            Y_train, Y_test = map(lambda data: data[:, y], (self.train_data, self.test_data))
            Y_train, Y_test = map(lambda x: x.reshape(-1, 1), (Y_train, Y_test))
            data = (X_train, Y_train, X_test, Y_test)
            tasks.append((y, len(X), data)) 

        n_jobs = self.args.n_jobs
        set_start_method('spawn', force=True)
        with Pool(n_jobs) as p:
            node_scores = list(tqdm(p.imap(self.train_one, tasks, chunksize=1), total=len(tasks), disable=not self.args.verbose))
            p.close()
            p.join()
        return node_scores

    def find_sink(self, variables):
        if len(variables) == 1:
            return variables[0]
        node_scores = self.train_parallel(variables) 
        sink_idx = np.argmin([x[1] for x in node_scores])
        sink_node = node_scores[sink_idx][0]
        if len(variables) == self.args.num_variables:
            self.g_inverse = [x[2] for x in node_scores]
        return sink_node

    # def pruning(self, threshold):
    #     _, d = self.data.shape 
    #     variables = list(range(self.args.num_variables))
    #     # train_res = self.train_parallel(variables)
    #     # g_inverse = [x[2] for x in train_res]
    #     A = np.ones((d, d))
    #     for i in range(1, d):
    #         gam = train_gam(self.data[:, self.causal_order[:i]], self.g_inverse[self.causal_order[i]], n_basis=10)
    #         p_values = gam.statistics_['p_values']
    #         print(f'{p_values=}')
    #         print(f'{self.causal_order[:i]=}')
    #         A[self.causal_order[:i], self.causal_order[i]] = p_values[:-1]
    #     A = (A < threshold) * 1 
    #     return A

def Ours(data, args, **kwargs):
    """
    Input: 
    data: numpy array with shape of (#samples, #num_variables)
    args: dictionary of arguments
    Output:
    adjacency_matrix
    """
    trainer = OursTrainer(args)
    data = torch.FloatTensor(data)
    dag, causal_order = trainer.fit(data, prunning=kwargs['prunning'])
    return dag, causal_order