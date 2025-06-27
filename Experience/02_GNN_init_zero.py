import sys
sys.path.append("../.")

import os
import time
import argparse
import pathlib
import json
import math
from copy import deepcopy

from pprint import pprint
from tqdm.auto import tqdm
import numpy as np

import torch
from torch import nn
from torch.nn import Module
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential

from torch_geometric.nn import MessagePassing
from torch_geometric.utils.isolated import remove_isolated_nodes

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.metrics.ml_metrics.external_metrics import mape_quantile
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation

from solver_utils import get_obs
from graph_utils import prepare_dataset, get_all_active_powers
from utils import NpEncoder, compute_local_conservation_error, compute_mean_std

def create_loaders(benchmark, device="cpu", batch_size=128, eval=False):
    
    train_loader, val_loader, test_loader, test_ood_loader = prepare_dataset(benchmark=benchmark, 
                                                                            batch_size=batch_size, 
                                                                            device=device,
                                                                            eval=eval)
    
    if not(eval):
        return train_loader, val_loader, test_loader, test_ood_loader
    else:
        return test_loader, test_ood_loader

##############################
# GNN layers
##############################

class GPGinput_without_NN(MessagePassing):
    """Graph Power Grid Input layer

    This is the input layer of GNN initialize the theta (voltage angles) with zeros and
    updates them through power flow equation

    """
    def __init__(self,
                 ref_node,
                 sn_mva,
                 device="cpu",
                 ):
        super().__init__(aggr="add")
        self.theta = None
        self.sn_mva = sn_mva
        self.device = device
        self.ref_node=ref_node

    def forward(self, batch, theta_init=None):
        
        # Initialize the voltage angles (theta) with zeros
        if theta_init is None:
            self.theta = torch.zeros_like(batch.y, dtype=batch.y.dtype)
        else:
            self.theta = theta_init

        # Compute a message and propagate it to each node, it does 3 steps
        # 1) It computes a message (Look at the message function below)
        # 2) It propagates the message using an aggregation (sum here)
        # 3) It calls the update function which could be Neural Network
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * self.sn_mva
                                 )
        n_bus = batch.ybus.size()[1]
        n_sub = n_bus / 2
        # keep only the diagonal elements of the ybus 3D tensors
        ybus = batch.ybus.view(-1, n_bus, n_bus) * self.sn_mva
        denominator = torch.hstack([ybus[i].diag() for i in range(len(ybus))]).reshape(-1,1)
        # ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        # denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        numerator = input_node_power - aggr_msg
        # out = (input_node_power - aggr_msg) / denominator
        indices = torch.where(denominator.flatten()!=0.)[0]
        out = torch.zeros_like(denominator)
        out[indices] = torch.divide(numerator[indices], denominator[indices])

        #we impose that reference node has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        #we impose the not used buses to have theta=0
        out[denominator==0] = 0
        
        # impose also the unused buses to have zero thetas
        
        
        return out, aggr_msg
    
    def message(self, y_j, edge_weights):
        """Compute the message that should be propagated
        
        This function compute the message (which is the multiplication of theta and 
        admittance matrix elements connecting node i to j)

        Args:
            y_j (_type_): the theta (voltage angle) value at a neighboring node j
            edge_weights (_type_): corresponding edge_weight (admittance matrix element)

        Returns:
            _type_: active powers for each neighboring node
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    def update(self, aggr_out):
        """update function of message passing layers

        We output directly the aggreated message (sum)

        Args:
            aggr_out (_type_): the aggregated message

        Returns:
            _type_: the aggregated message
        """
        return aggr_out
    
class GPGintermediate(MessagePassing):
    def __init__(self,
                 ref_node,
                 sn_mva=1.0,
                 device="cpu"):
        super().__init__(aggr="add")
        self.theta = None
        self.ref_node = ref_node
        self.sn_mva = sn_mva
        self.device = device
        
    
    def forward(self, batch, theta):
        self.theta = theta
        
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=self.theta,
                                  edge_weights=batch.edge_attr_no_diag * self.sn_mva
                                 )

        n_bus = batch.ybus.size()[1]
        # keep only the diagonal elements of the ybus 3D tensors for denominator part
        ybus = batch.ybus.view(-1, n_bus, n_bus) * self.sn_mva
        # Keeping only the diagonal elements y_ii for denominator part
        # ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        # denominator = ybus[ybus.nonzero(as_tuple=True)].view(-1,1)
        denominator = torch.hstack([ybus[i].diag() for i in range(len(ybus))]).reshape(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        numerator = input_node_power - aggr_msg
        # out = (input_node_power - aggr_msg) / denominator
        indices = torch.where(denominator.flatten()!=0.)[0]
        out = torch.zeros_like(denominator)
        out[indices] = torch.divide(numerator[indices], denominator[indices])

        #we impose that reference node has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        #we impose the not used buses to have theta=0
        out[denominator==0] = 0
        
        return out, aggr_msg

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp

    def update(self, aggr_out):
        return aggr_out

class LocalConservationLayer(MessagePassing):
    """
    Same as power equilibrium equations
    
    It can be also injected as inputs to the next layer
    """
    def __init__(self, sn_mva=1.0):
        super().__init__(aggr="add")
        self.thetas = None
        self.sn_mva = sn_mva
        
    def forward(self, batch, thetas=None):
        # thetas from the previous GNN layer
        self.thetas = thetas

        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * self.sn_mva)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        # compute the local conservation error (node level)
        nodal_error = input_node_power - aggr_message

        return nodal_error

    def message(self, y_i, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
######################
# Model definition
######################

class GPGmodel_without_NN(torch.nn.Module):
    """Create a Graph Power Grid (GPG) model without learning
    """
    def __init__(self,
                 ref_node,
                 sn_mva,
                 num_gnn_layers=10,
                 device="cpu"):
        super().__init__()
        self.ref_node = ref_node
        self.sn_mva = sn_mva
        self.num_gnn_layers = num_gnn_layers
        self.device = device

        self.input_layer = None
        self.lc_layer = None
        self.inter_layers = None

        self.build_model()

    def build_model(self):
        """Build the GNN message passing model

        It composed of a first input layer and a number of intermediate message passing layers
        These layes interleave with local conservation layers which allow to compute the error
        at the layer level
        """
        self.input_layer = GPGinput_without_NN(ref_node=self.ref_node, sn_mva=self.sn_mva, device=self.device)
        self.lc_layer = LocalConservationLayer(sn_mva=self.sn_mva)
        self.inter_layers = torch.nn.ModuleList([GPGintermediate(ref_node=self.ref_node, 
                                                                 sn_mva=self.sn_mva,
                                                                 device=self.device) 
                                                 for _ in range(self.num_gnn_layers)])

    def forward(self, batch, theta_init=None):
        errors = []
        out, _ = self.input_layer(batch, theta_init=theta_init)
        nodal_error = self.lc_layer(batch, out)
        errors.append(abs(nodal_error).sum())
        
        for gnn_layer in self.inter_layers:
            out, _ = gnn_layer(batch, out)
            nodal_error = self.lc_layer(batch, out)
            errors.append(abs(nodal_error).sum())

        return out, errors

def optimize(model, 
             data_loader,
             opt_num=0,
             save_path=None,
             model_name="model.pt"
             ):
    predictions_list = []
    observations_list = []
    error_per_batch = []
    total_time = 0
    for batch in tqdm(data_loader):
        beg_ = time.perf_counter()
        out, errors = model(batch)
        end_ = time.perf_counter()
        total_time += end_ - beg_
        predictions_list.append(out)
        observations_list.append(batch.y)
        #error_per_batch.append(torch.mean(torch.vstack(errors)))
        error_per_batch.append([float(error.detach().cpu().numpy()) for error in errors])
    observations = torch.vstack(observations_list)
    predictions = torch.vstack(predictions_list)
    #errors = np.vstack([error.cpu().numpy() for error in error_per_batch])
    errors = np.vstack(error_per_batch)
    errors = errors.mean(axis=0)
    
    if save_path is not None:
        train_save_path = pathlib.Path(save_path) / f"opt_{opt_num}"
        if not train_save_path.exists():
                train_save_path.mkdir(parents=True)
        model_path = train_save_path / f"{model_name}"
        torch.save({
            "errors": errors,
            "total_time": total_time,
            "predictions": predictions,
            "observations": observations
        }, model_path)
    return observations, predictions, predictions_list
        

def evaluate(model, data_loader, benchmark, dataset="_test_dataset", opt_num=0, save_path=None, model_name="model.pt"):
    
    observations, predictions, predictions_list = optimize(model, data_loader, opt_num=opt_num, save_path=save_path, model_name=model_name)
    
    batch = next(iter(data_loader))
    beg_ = time.perf_counter()
    _, _ = model(batch)
    end_ = time.perf_counter()
    time_inf = end_ - beg_
    
    predictions = predictions * (180/math.pi)
    
    env, obs = get_obs(benchmark)
    sn_mva = env.backend._grid.get_sn_mva()
    my_predictions = {}
    my_predictions["p_or"], my_predictions["p_ex"] = get_all_active_powers(getattr(benchmark, dataset).data,
                                                                           obs,
                                                                           theta_bus=predictions.view(-1,obs.n_sub*2).cpu().numpy(),
                                                                           sn_mva=sn_mva)
    
    evaluation = PowerGridEvaluation.from_benchmark(benchmark)
    metrics = evaluation.evaluate(observations=getattr(benchmark, dataset).data, 
                                  predictions=my_predictions, 
                                  env=env)
    
    metrics = compute_local_conservation_error(data_loader, predictions_list, sn_mva, metrics, obs)
    
    predictions = predictions.detach().cpu().numpy()
    observations = observations.detach().cpu().numpy()
    
    metrics["theta"] = {}
    mape10 = mape_quantile(y_true=observations, y_pred=predictions, quantile=0.9)
    mape90 = mape_quantile(y_true=observations, y_pred=predictions, quantile=0.1)
    mae = np.mean(np.abs(observations - predictions))
    wmape = np.mean(np.abs(predictions - observations)) / np.mean(np.abs(observations))
    metrics["theta"]["MAPE10"] = mape10
    metrics["theta"]["MAPE90"] = mape90
    metrics["theta"]["MAE"] = mae
    metrics["theta"]["WMAPE"] = wmape
    
    metrics["INF_TIME"] = time_inf
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_name", help="The name of the environemnt", default="l2rpn_case14_sandbox", required=False)
    parser.add_argument("-nt", "--n_opt", help="if True it trains the model, otherwise evaluate the model", required=False, default=1, type=int)
    parser.add_argument("-p", "--load_path", help="the path from which the model should be loaded", required=False, default=None, type=str)
    parser.add_argument("-mn", "--model_name", help="the path from which the model should be loaded", required=False, default="model.pt", type=str)
    parser.add_argument("-s", "--save_path", help="Path where the model should be saved", required=False, default="trained_model/GNN_init_zero_14", type=str)
    parser.add_argument("-ng", "--num_gnn_layers", help="Number of GNN layers", required=False, default="20", type=int)
    # parser.add_argument("-hl", "--hidden_layers", help="The set of hidden layers for MLP", required=False, default="(800, 800, 500, 300, 300)", type=str)
    parser.add_argument("-bs", "--batch_size", help="Batch size", required=False, default=512, type=int)
    parser.add_argument("-rn", "--ref_node", help="reference node in the graph", required=True, default=0, type=int) # 35 for neurips_small
    args = parser.parse_args()
    
    ENV_NAME = args.env_name

    PATH = pathlib.Path().resolve().parent
    BENCH_CONFIG_PATH = PATH / "configs" / (ENV_NAME + ".ini")
    DATA_PATH = PATH / "Datasets" / ENV_NAME / "DC" / "Benchmark4_complete"
    LOG_PATH = PATH / "lips_logs.log"
    
    # print(type(args.hidden_layers))
    # hidden_layers = eval(args.hidden_layers)
    print(f"Env used: {ENV_NAME}")
    # print(f"hidden_layers: {hidden_layers}")
    print(f"Number of GNN layers: {args.num_gnn_layers}")
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # or "cuda:0" if you have any GPU
    batch_size = args.batch_size
    
    print(f"Device: {device}")
    print(f"batch_size: {batch_size}")
    
    benchmark = PowerGridBenchmark(dataset_path=DATA_PATH,
                                   benchmark_name="Benchmark4",
                                   load_data_set=True,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=LOG_PATH)
    
    env, obs = get_obs(benchmark)
    n_bus = env.n_sub * 2
    sn_mva = env.backend._grid.get_sn_mva()
    
    test_loader, test_ood_loader = create_loaders(benchmark=benchmark, 
                                                  batch_size=batch_size, 
                                                  device=device,
                                                  eval=True)
    
    metrics_all = []
    # metrics_all["test"] = {}
    # metrics_all["test_ood"] = {}
    for opt_num in range(args.n_opt):
        model = GPGmodel_without_NN(ref_node=args.ref_node, sn_mva=sn_mva, num_gnn_layers=args.num_gnn_layers, device=device).to(device)
        
            
        # model, train_losses, val_losses = optimize(model,
        #                                             train_loader=train_loader,
        #                                             save_freq=50,
        #                                             save_path=args.save_path,
        #                                             sn_mva=sn_mva,
        #                                             train_continue=eval(args.train_continue),
        #                                             load_path=args.load_path,
        #                                             model_name=args.model_name,
        #                                             train_num=train_num                                                        
        #                                             )
        
            
        metrics = {}
        metrics["test"] = evaluate(model, test_loader, benchmark, dataset="_test_dataset", opt_num=opt_num, save_path=args.save_path, model_name="model_test.pt")
        metrics["test_ood"] = evaluate(model, test_ood_loader, benchmark, dataset="_test_ood_topo_dataset", opt_num=opt_num, save_path=args.save_path, model_name="model_ood.pt")
        
        metrics_all.append(metrics)
        
        with open(os.path.join(args.save_path, f'result_{opt_num}.json'), "w", encoding="utf-8") as f:
            json.dump(obj=metrics, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
    
    compute_mean_std(metrics_all, save_path=args.save_path)
    