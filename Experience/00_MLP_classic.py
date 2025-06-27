import sys
sys.path.append("../.")

import os
import time
import argparse
import pathlib
from pprint import pprint
import copy
import json
from copy import deepcopy
from pprint import pprint

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from lips.metrics.ml_metrics.external_metrics import mape_quantile

from solver_utils import get_obs
from graph_utils import prepare_dataset, get_all_active_powers
from utils import NpEncoder, compute_local_conservation_error_mlp, compute_mean_std

class PowerGridDataset(Dataset):
    def __init__(self, benchmark, dataset:str, device="cpu"):
        super().__init__()
        # self.x_attr = benchmark.config.get_option("attr_x")
        # self.y_attr = benchmark.config.get_option("attr_y")
        self.tau_attr = benchmark.config.get_option("attr_tau")
        self.dataset = getattr(benchmark, dataset)
        self.device = device
        
        # self.features = torch.cat([torch.tensor(self.dataset.data[attr], device=self.device) for attr in (*self.x_attr, *self.tau_attr)], dim=1)
        self.features = torch.cat([torch.tensor(self.dataset.data[attr], device=self.device) for attr in self.tau_attr], dim=1)
        # self.targets = torch.cat([torch.tensor(self.dataset.data[attr], device=self.device) for attr in self.y_attr if attr in ("theta_or", "theta_ex")], dim=1)
                
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]#, self.targets[index]
    
    def __iter__(self):
        # print(range(len(self)))
        # for i in range(len(self)):
        #     print(i)
        #     yield self.features[i], self.targets[i]
        self.i = 0
        return self
            
    def __next__(self):
        if self.i < len(self):
            tmp = self.i
            self.i += 1
            print(tmp)
            return self.features[tmp]#, self.targets[tmp]
        else:
            raise StopIteration

class WrappedDataLoader:
    def __init__(self, dl, benchmark_dl, n_bus=28):
        self.dl = dl
        self.benchmark_dl = benchmark_dl
        self.n_bus = n_bus
        #self.batch_size = self.dl.batch_size
        
        self.stop = len(dl)
        self.current = 0
        
        self.__reset_lists()
        self.reset()
        
    def preprocess(self, topo_inf, batch_dl):
        self.__reset_lists()
        for idx in list(range(0, batch_dl.size(0), self.n_bus)):
            self.features.append(torch.flatten(batch_dl.x[idx: idx+self.n_bus]))
            self.targets.append(torch.flatten(batch_dl.y[idx: idx+self.n_bus]))
        return torch.cat([torch.vstack(self.features), topo_inf], dim=1), torch.vstack(self.targets), batch_dl
    
    def reset(self):
        self.dl_it = iter(self.dl)
        self.benchmark_dl_it = iter(self.benchmark_dl)
        
    def __reset_lists(self):
        self.features = list()
        self.targets = list()
        self.topo_vect = list()
        self.line_status = list()
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += 1
        
        if self.current <= self.stop:
            batch_benchmark_dl = next(self.benchmark_dl_it)
            batch_dl = next(self.dl_it)
            return self.preprocess(batch_benchmark_dl, batch_dl)
            # return self.preprocess(None, batch_dl)
        else:
            self.reset()
            self.current = 0
            raise StopIteration

def create_loaders(benchmark, n_bus, device="cpu", batch_size=128, eval=False):
    
    train_loader, val_loader, test_loader, test_ood_loader = prepare_dataset(benchmark=benchmark, 
                                                                            batch_size=batch_size, 
                                                                            device=device,
                                                                            eval=eval)
    
    if not(eval):
        benchmark_train_loader = DataLoader(PowerGridDataset(benchmark, "train_dataset", device=device), batch_size=batch_size)
        benchmark_val_loader = DataLoader(PowerGridDataset(benchmark, "val_dataset", device=device), batch_size=batch_size)
        
        train_dl = WrappedDataLoader(train_loader, benchmark_train_loader, n_bus=n_bus)
        val_dl = WrappedDataLoader(val_loader, benchmark_val_loader, n_bus=n_bus)
    
    benchmark_test_loader = DataLoader(PowerGridDataset(benchmark, "_test_dataset", device=device), batch_size=batch_size)
    test_dl = WrappedDataLoader(test_loader, benchmark_test_loader, n_bus=n_bus)
    
    benchmark_test_ood_loader = DataLoader(PowerGridDataset(benchmark, "_test_ood_topo_dataset", device=device), batch_size=batch_size)
    test_ood_dl = WrappedDataLoader(test_ood_loader, benchmark_test_ood_loader, n_bus=n_bus)
    
    if not(eval):
        return train_dl, val_dl, test_dl, test_ood_dl
    else:
        return test_dl, test_ood_dl

class LocalConservationLayer(MessagePassing):
    """Compute local conservation error

    This class computes the local conservation error without any update of voltage angles.

    Args:
        MessagePassing (_type_): _description_
    """
    def __init__(self):
        super().__init__(aggr="add")
        self.thetas = None
        
    def forward(self, batch, thetas=None, sn_mva=1.0):
        # theta from previous GNN layer
        self.thetas = thetas

        # The difference with GPG layers resides also in propagation which gets the edge_index
        # with self loops (with diagonal elements of adjacency matrix)
        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * sn_mva)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        # compute the local conservation error (at node level)
        nodal_error = input_node_power - aggr_message

        return nodal_error

    def message(self, y_i, y_j, edge_weights):
        """
        Compute the message
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

def infer_dim(obs):
    n_bus = obs.n_sub * 2
    input_dim = n_bus + n_bus + obs.n_line + len(obs.topo_vect)# n_bus_prod + n_bus_load + n_line_status + n_topo_vect
    output_dim = n_bus# n_bus_theta
    return input_dim, output_dim

class FullyConnected(torch.nn.Module):
    def __init__(self, input_dim:int = 133, hidden_layers: tuple=(150, 80), output_dim=28):
        super().__init__()
        
        self.layers = torch.nn.Sequential()
        for index, (layer_c, layer_n) in enumerate(zip((input_dim, *hidden_layers), (*hidden_layers, output_dim))):
            self.layers.append(torch.nn.Linear(in_features=layer_c, out_features=layer_n))
            if index < len(hidden_layers):
                self.layers.append(torch.nn.ReLU())
                
    def forward(self, xb):
        return self.layers(xb)


def train_model(model, 
                train_loader, 
                val_loader, 
                optimizer, 
                loss_fn,
                epochs=100, 
                scheduler=None,
                save_freq=20,
                save_path=None,
                train_continue=False,
                load_path=None,
                model_name=None,
                train_num=1):
    
    if train_continue:
        if (args.load_path is None) or not(os.path.exists(args.load_path)):
            raise FileNotFoundError("The load_path is not indicated or not existing")
        checkpoint = torch.load(os.path.join(load_path, f"train_{train_num}", model_name), weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        min_epochs = epoch
    else:
        min_epochs = 0
        train_losses = []
        val_losses = []
    
    best_model_state = model.state_dict()
    best_optimizer_state = optimizer.state_dict()
    old_val_loss = 1e9
    
    for epoch in range(min_epochs, epochs):
        model.train()
        epoch_loss = 0
        
        for features, targets, batch in train_loader:
            preds = model(features)
            loss = loss_fn(preds.view(targets.size()), targets)
            epoch_loss += loss.item()            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss /= len(train_loader)
        
        train_losses.append(epoch_loss)        
        print_string = f"train loss: {epoch_loss:>10.4f}"
            
        if val_loader is not None:
            val_loss = eval_model(model, val_loader, loss_fn)
            val_losses.append(val_loss)
            print_string += f" | val loss: {val_loss:>10.4f}"
                
        if scheduler is not None:
            scheduler.step(val_loss)
            
        print_string += f" | epochs: [{epoch:>4d}\{epochs:4d}]"
        
        print(print_string)
        
        if save_path is not None:
            train_save_path = pathlib.Path(save_path) / f"train_{train_num}"
            if not train_save_path.exists():
                train_save_path.mkdir(parents=True)
            model_path = train_save_path / f"model_epoch_{str(epoch+1)}.pt"
            if (((epoch+1) % save_freq) == 0) & (val_loss < old_val_loss):
                old_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                best_optimizer_state = deepcopy(optimizer.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer_state,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, model_path)
        
    return model, train_losses, val_losses

def eval_model(model, data_loader, loss_fn):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for features, targets, batch in data_loader:
            preds = model(features)
            loss = loss_fn(preds.view(targets.size()), targets)
            eval_loss += loss.item()
        eval_loss /= len(data_loader)
        
    return eval_loss

def predict(model, data_loader):
    predictions = []
    observations = []
    model.eval()
    
    with torch.no_grad():
        for features, targets, _ in data_loader:
            predictions.append(model(features))
            observations.append(targets)
        
    return torch.vstack(predictions), torch.vstack(observations), predictions

def evaluate(model, data_loader, benchmark, dataset="_test_dataset", sn_mva=1.0):
    model.eval()
    beg_ = time.perf_counter()
    predictions, observations, predictions_list = predict(model, data_loader)
    end_ = time.perf_counter()
    time_inf = end_ - beg_
    
    predictions = predictions.cpu().numpy()
    observations = observations.cpu().numpy()
        
    env, obs = get_obs(benchmark)

    pred_dict = {}
    beg_ = time.perf_counter()
    p_ors_pred, p_exs_pred = get_all_active_powers(getattr(benchmark, dataset).data, 
                                                   obs, 
                                                   theta_bus=predictions,
                                                   sn_mva=sn_mva)
    end_ = time.perf_counter()
    time_post_proc = end_ - beg_
    pred_dict["p_or"] = p_ors_pred
    pred_dict["p_ex"] = p_exs_pred


    evaluation = PowerGridEvaluation.from_benchmark(benchmark)
    metrics = evaluation.evaluate(observations=getattr(benchmark, dataset).data, 
                                  predictions=pred_dict, 
                                  env=env)
    
    metrics = compute_local_conservation_error_mlp(data_loader, predictions_list, sn_mva, metrics, obs)
    
    metrics["theta"] = {}
    mape10 = mape_quantile(y_true=observations, y_pred=predictions, quantile=0.9)
    mape90 = mape_quantile(y_true=observations, y_pred=predictions, quantile=0.1)
    mae = np.mean(np.abs(observations - predictions))
    wmape = np.mean(np.abs(predictions - observations)) / np.mean(np.abs(observations))
    # print("MAPE10 on theta: ", MAPE_10)
    metrics["theta"]["MAPE10"] = mape10
    metrics["theta"]["MAPE90"] = mape90
    metrics["theta"]["MAE"] = mae
    metrics["theta"]["WMAPE"] = wmape
    
    metrics["INF_TIME"] = time_inf
    metrics["POST_PROC_TIME"] = time_post_proc    
    
    print("INF_TIME: ", {time_inf})
    print("POST_PROC_TIME: ", {time_post_proc})
    
    
    print(f"******* {dataset} ********")
    pprint(metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", help="The name of the environemnt", default="l2rpn_case14_sandbox", required=False)
    parser.add_argument("-t", "--train", help="if True it trains the model, otherwise evaluate the model", default="False")
    parser.add_argument("-nt", "--n_train", help="if True it trains the model, otherwise evaluate the model", required=False, default=1, type=int)
    parser.add_argument("-p", "--load_path", help="the path from which the model should be loaded", required=False, default=None, type=str)
    parser.add_argument("-mn", "--model_name", help="the path from which the model should be loaded", required=False, default="model_epoch_200.pt", type=str)
    parser.add_argument("-s", "--save_path", help="Path where the model should be saved", required=False, default="trained_model/MLP_14", type=str)
    parser.add_argument("-hl", "--hidden_layers", help="The set of hidden layers for MLP", required=False, default="(800, 800, 500, 300, 300)", type=str)
    parser.add_argument("-ep", "--epochs", help="Number of training epochs", required=False, default=200, type=int)
    parser.add_argument("-bs", "--batch_size", help="Batch size", required=False, default=128, type=int)
    parser.add_argument("-tc", "--train_continue", help="if continue the training", required=False, default="False", type=str)
    args = parser.parse_args()
    
    # ENV_NAME = "l2rpn_case14_sandbox"
    # ENV_NAME = "l2rpn_neurips_2020_track1_small"
    ENV_NAME = args.env_name

    PATH = pathlib.Path().resolve().parent
    BENCH_CONFIG_PATH = PATH / "configs" / (ENV_NAME + ".ini")
    DATA_PATH = PATH / "Datasets" / ENV_NAME / "DC" / "Benchmark4_complete"
    LOG_PATH = PATH / "lips_logs.log"
    
    # print(type(args.hidden_layers))
    hidden_layers = eval(args.hidden_layers)
    print(f"Env used: {ENV_NAME}")
    print(f"Train mode: {args.train}")
    print(f"hidden_layers: {hidden_layers}")
    print(f"Epochs: {args.epochs}")
    if args.load_path is not None:
        print(f"Load path: {args.load_path}")
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # or "cuda:0" if you have any GPU
    batch_size = args.batch_size
    
    benchmark = PowerGridBenchmark(dataset_path=DATA_PATH,
                                   benchmark_name="Benchmark4",#"DoNothing",
                                   load_data_set=True,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=LOG_PATH)
    
    env, obs = get_obs(benchmark)
    n_bus = env.n_sub * 2
    sn_mva = env.backend._grid.get_sn_mva()
    
    if eval(args.train):
        loaders = create_loaders(benchmark=benchmark, n_bus=n_bus, device=device, batch_size=batch_size)
        train_loader, val_loader, test_loader, test_ood_loader = loaders
    
    input_dim, output_dim = infer_dim(obs)
    
    metrics_all = []
    # loaders = create_loaders(benchmark_name="Benchmark4", device=device, batch_size=batch_size)
    # train_loader, val_loader, test_loader, test_ood_loader = loaders
    for train_num in range(args.n_train):
        model = FullyConnected(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        loss_fn = torch.nn.MSELoss()
        
        if eval(args.train):
            print("Enter the training mode")
            model.train()
            
            model, train_losses, val_losses = train_model(model, 
                                                        train_loader, 
                                                        val_loader, 
                                                        optimizer, 
                                                        loss_fn, 
                                                        epochs=args.epochs, 
                                                        scheduler=scheduler,
                                                        save_freq=20,
                                                        save_path=args.save_path,
                                                        train_continue=eval(args.train_continue),
                                                        load_path=args.load_path,
                                                        model_name=args.model_name,
                                                        train_num=train_num
                                                        )
            
        else:
            print("Enter the evaluation mode")
            if (args.load_path is None) or not(os.path.exists(args.load_path)):
                raise FileNotFoundError("The load_path is not indicated or not existing")
            
            loaders = create_loaders(benchmark=benchmark, n_bus=n_bus, device=device, batch_size=batch_size, eval=True)
            test_loader, test_ood_loader = loaders
            
            checkpoint = torch.load(os.path.join(args.load_path, f"train_{train_num}", args.model_name), weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_losses = checkpoint["train_losses"]
            val_losses = checkpoint["val_losses"]
        
        metrics = {}    
        metrics["test"] = evaluate(model, test_loader, benchmark,  dataset="_test_dataset", sn_mva=sn_mva)
        metrics["test_ood"] = evaluate(model, test_ood_loader, benchmark, dataset="_test_ood_topo_dataset", sn_mva=sn_mva)

        metrics_all.append(metrics)
        
        with open(os.path.join(args.save_path, f'result_{train_num}.json'), "w", encoding="utf-8") as f:
            json.dump(obj=metrics, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
            
    compute_mean_std(metrics_all, save_path=args.save_path)
