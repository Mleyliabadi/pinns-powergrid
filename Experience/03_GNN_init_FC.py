import sys
sys.path.append("../.")

import os
import time
import argparse
import pathlib
import math
import copy
import json
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
import torch_geometric.transforms as T
from torch_geometric.utils.isolated import remove_isolated_nodes

from lips.benchmark.powergridBenchmark import PowerGridBenchmark
from lips.evaluation.powergrid_evaluation import PowerGridEvaluation
from lips.metrics.ml_metrics.external_metrics import mape_quantile

from solver_utils import get_obs
from graph_utils import prepare_dataset
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

class GPGinput(MessagePassing):
    def __init__(self,
                 ref_node,
                 sn_mva=1.0,
                 device="cpu"
                 ):
        super().__init__(aggr="add")
        self.ref_node = ref_node
        self.sn_mva = sn_mva
        self.device = device
        

    def forward(self, batch, init_theta):
        """The forward pass of the Input Layer

        The input layer 
        Parameters
        ----------
        init_theta : _type_
            the initial_thetas
        batch : _type_
            the batch including various information concerning the grid
        sn_mva : float, optional
            the reference power value, by default 1.0

        Returns
        -------
        _type_
            the new theta values
        """
        # init the thetas with zeros 
        # init_theta = torch.zeros(batch.y.size(), device=self.device)
        
        # propagate and aggregate the messages from neighboring nodes
        aggr_msg = self.propagate(batch.edge_index_no_diag,
                                  y=init_theta,
                                  edge_weights=batch.edge_attr_no_diag * self.sn_mva
                                 )
        
        # keep only the diagonal elements of the ybus 3D tensors
        n_bus = batch.ybus.size()[1]
        # n_sub = n_bus / 2
        ybus = batch.ybus.view(-1, n_bus, n_bus) * self.sn_mva
        # keeping only the diagonal of ybus matrices (y_ii)
        # ybus = ybus * torch.eye(*ybus.shape[-2:], device=self.device).repeat(ybus.shape[0], 1, 1)
        # the diag function do the same job as the above line
        denominator = torch.hstack([ybus[i].diag() for i in range(len(ybus))]).reshape(-1,1)
        
        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        numerator = input_node_power - aggr_msg
        
        indices = torch.where(denominator.flatten()!=0.)[0]
        out = torch.zeros_like(denominator)
        out[indices] = torch.divide(numerator[indices], denominator[indices])

        #we impose that node 0 has theta=0
        out = out.view(-1, n_bus, 1) - out.view(-1,n_bus,1)[:,self.ref_node].repeat_interleave(n_bus, 1).view(-1, n_bus, 1)
        out = out.flatten().view(-1,1)
        
        # imposing theta=0 for the bus which are not used
        out[denominator==0] = 0
        
        return out, aggr_msg
    
    def message(self, y_j, edge_weights):
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    def update(self, aggr_out):
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


class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, n_input, n_output, n_hidden, n_layers, activation=nn.ReLU):
        super().__init__()
        activation = activation
        self.fcs = nn.Sequential(*[
                        nn.Linear(n_input, n_hidden),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(n_hidden, n_hidden),
                            activation()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(n_hidden, n_output)       

    def forward(self, batch):
        out = batch.x
        out = self.fcs(out)
        out = self.fch(out)
        out = self.fce(out)
        return out
    
class FullyConnected(torch.nn.Module):
    def __init__(self, input_dim:int = 2, hidden_layers: tuple=(150, 80), output_dim=1):
        super().__init__()
        
        self.layers = torch.nn.Sequential()
        for index, (layer_c, layer_n) in enumerate(zip((input_dim, *hidden_layers), (*hidden_layers, output_dim))):
            self.layers.append(torch.nn.Linear(in_features=layer_c, out_features=layer_n))
            if index < len(hidden_layers):
                self.layers.append(torch.nn.ReLU())
                
    def forward(self, batch):
        out = batch.x
        return self.layers(out)
    
######################
# Model definition
######################

class GPGmodel(Module):
    def __init__(self,
                 ref_node,
                 input_dim=2,
                 output_dim=1,
                #  embedding_size=16,
                 num_gnn_layers=10,
                 sn_mva=1.0,
                 device="cpu"):
        super().__init__()
        self.ref_node = ref_node
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.embedding_size = embedding_size
        self.num_gnn_layers = num_gnn_layers
        
        self.device = device

        # self.embedding_layer = None
        self.input_layer = None
        self.lc_layer = None
        self.inter_layers = None
        self.decoding_layer = None
        
        self.sn_mva = sn_mva
        
        self.build_model()

    def build_model(self):
        # self.embedding_layer = Linear(self.input_dim, self.output_dim)
        # self.embedding_layer = FCN(n_input=self.input_dim, n_output=self.output_dim, n_hidden=50, n_layers=8)
        self.embedding_layer = FullyConnected(input_dim=self.input_dim, output_dim=self.output_dim, hidden_layers=(100,80,50,30))
        self.input_layer = GPGinput(ref_node=self.ref_node, sn_mva=self.sn_mva, device=self.device)
        # self.lc_layer = LocalConservationLayer(sn_mva=self.sn_mva)
        self.inter_layer = GPGintermediate(ref_node=self.ref_node, sn_mva=self.sn_mva, device=self.device)
        #self.inter_layers = ModuleList([GPGintermediate(device=self.device) for _ in range(self.num_gnn_layers)])
        # self.decoding_layer = Sequential(Linear(self.output_dim, 10),
        #                                  ReLU(),
        #                                  Linear(10, self.output_dim)).to(self.device)
        #self.decoding_layer = Linear(self.output_dim, self.output_dim)

    def forward(self, batch):
        # errors = []
        init_theta = self.embedding_layer(batch) # start by NN init
        # init_theta = torch.zeros_like(batch.y, dtype=batch.y.dtype) # flat start
        out, _ = self.input_layer(batch, init_theta)
        # nodal_error = self.lc_layer(batch, out)
        # errors.append(abs(nodal_error).sum())
        
        #for gnn_layer, lc_layer_ in zip(self.inter_layers, itertools.repeat(self.lc_layer)):
        for _ in range(self.num_gnn_layers):
            out, _ = self.inter_layer(batch, out)
            #nodal_error = self.lc_layer(batch, out)
            #errors.append(abs(nodal_error).sum())
            
        # out = self.decoding_layer(out)

        # return out, errors
        return out, None

def train_model(model,
                train_loader,
                optimizer,
                val_loader=None,
                loss_fn=nn.MSELoss(),
                epochs=100,
                scheduler=None,
                save_freq=20,
                save_path=None,
                sn_mva=1.0,
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
        train_losses_data = checkpoint["train_losses_data"]
        train_losses_physics = checkpoint["train_losses_physics"]
        val_losses = checkpoint["val_losses"]
        min_epochs = epoch
    else:
        min_epochs = 0
        train_losses = []
        train_losses_data = []
        train_losses_physics = []
        val_losses = []
    best_model_state = model.state_dict()
    best_optimizer_state = optimizer.state_dict()
    old_val_loss = 1e9
    lc_layer = LocalConservationLayer(sn_mva=sn_mva)
    # transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])

    # for epoch in tqdm(range(epochs)):
    for epoch in range(min_epochs, epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_data = 0
        epoch_loss_physics = 0
        
        for batch in train_loader:
            # removing the isolated nodes for computation of data loss
            _, _, mask = remove_isolated_nodes(batch.edge_index)
            mask_tot = torch.zeros_like(batch.y, dtype=torch.bool).flatten()
            mask_tot[:len(mask)] = mask
            
            pred, error = model(batch)
            
            # take the last error from the last Local Conservation layer
            # loss_physics = error[-1]
            loss_physics = lc_layer(batch, pred)
            loss_physics = torch.mean((loss_physics - 0)**2)
            
            # compute the loss on data
            # loss_data = loss_fn(pred, batch.y)
            loss_data = torch.mean((pred[mask_tot] * (180/math.pi) - batch.y[mask_tot])**2)
            
            loss = loss_physics + loss_data
            
            epoch_loss += loss.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_physics += loss_physics.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #total_loss += (loss * len(batch.x))
            #total_loss += loss / 14
            #total_loss /= len(batch)
        epoch_loss /= len(train_loader)
        epoch_loss_data /= len(train_loader)
        epoch_loss_physics /= len(train_loader)
        # total_loss = total_loss.item() / len(train_loader.dataset)
        
        # TODO: add the physics and data losses to save 
        train_losses.append(epoch_loss)
        train_losses_data.append(epoch_loss_data)
        train_losses_physics.append(epoch_loss_physics)
        
        print_string = f"train loss: {epoch_loss:>10.4f}"
        
        if val_loader is not None:
            val_loss = eval_model(model, val_loader, loss_fn, lc_layer)
            val_losses.append(val_loss)
            print_string += f" | val loss: {val_loss:>10.4f}"
            
        print_string += f" | data loss: {epoch_loss_data:>10.4f} | physics loss: {epoch_loss_physics:>10.4f}"
        
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
                    'train_losses_data': train_losses_data,
                    'train_losses_physics': train_losses_physics,
                    'val_losses': val_losses
                }, model_path)

    return model, train_losses, val_losses

def eval_model(model, loader, loss_fn, lc_layer):
    model.eval()
    eval_loss = 0
    # predictions = list()
    # observations = list()
    

    with torch.no_grad():
        for batch in loader:
            # removing the isolated nodes for computation of data loss
            _, _, mask = remove_isolated_nodes(batch.edge_index)
            mask_tot = torch.zeros_like(batch.y, dtype=torch.bool).flatten()
            mask_tot[:len(mask)] = mask
            
            data = batch.x
            pred, errors = model(batch)            
            # loss_physics = errors[-1]
            loss_physics = lc_layer(batch, pred)
            loss_physics = torch.mean((loss_physics - 0)**2)
            
            # loss_data = loss_fn(pred, batch.y)
            loss_data = torch.mean((pred[mask_tot] * (180/math.pi) - batch.y[mask_tot])**2)
            
            loss = loss_physics + loss_data
            eval_loss += loss.item()
        eval_loss /= len(loader)
    # predictions = torch.vstack(predictions)
    # observations = torch.vstack(observations)
    # mean_loss = total_loss.item() / len(loader.dataset)
    # #total_constraint = total_constraint.item() / len(loader.dataset)
    
    return eval_loss

def predict(model, data_loader):
    predictions = []
    observations = []
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            preds, _ = model(batch)
            predictions.append(preds)
            observations.append(batch.y)
        
    return torch.vstack(predictions), torch.vstack(observations), predictions


def evaluate(model, data_loader, benchmark, dataset="_test_dataset", sn_mva=1.0):
    model.eval()
    beg_ = time.perf_counter()
    predictions, observations, predictions_list = predict(model, data_loader)
    end_ = time.perf_counter()
    time_inf = end_ - beg_
    
    # predictions = predictions.cpu().numpy()
    # observations = observations.cpu().numpy()
    
    
    from graph_utils import get_all_active_powers
    
    env, obs = get_obs(benchmark)

    pred_dict = {}
    beg_ = time.perf_counter()
    predictions *= (180/math.pi)
    p_ors_pred, p_exs_pred = get_all_active_powers(getattr(benchmark, dataset).data, 
                                                   obs, 
                                                   theta_bus=predictions.view(-1, obs.n_sub*2).cpu().numpy(),
                                                   sn_mva=sn_mva)
    end_ = time.perf_counter()
    time_post_proc = end_ - beg_
    pred_dict["p_or"] = p_ors_pred
    pred_dict["p_ex"] = p_exs_pred


    evaluation = PowerGridEvaluation.from_benchmark(benchmark)
    metrics = evaluation.evaluate(observations=getattr(benchmark, dataset).data, 
                                  predictions=pred_dict, 
                                  env=env)
    
    metrics = compute_local_conservation_error(data_loader, predictions_list, sn_mva, metrics, obs)
    
    observations = observations.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    
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
    parser.add_argument("-s", "--save_path", help="Path where the model should be saved", required=False, default="trained_model/GNN_init_FC_14", type=str)
    parser.add_argument("-ng", "--num_gnn_layers", help="Number of GNN layers", required=False, default="20", type=int)
    # parser.add_argument("-hl", "--hidden_layers", help="The set of hidden layers for MLP", required=False, default="(800, 800, 500, 300, 300)", type=str)
    parser.add_argument("-ep", "--epochs", help="Number of training epochs", required=False, default=100, type=int)
    parser.add_argument("-bs", "--batch_size", help="Batch size", required=False, default=512, type=int)
    parser.add_argument("-tc", "--train_continue", help="if continue the training", required=False, default="False", type=str)
    parser.add_argument("-tn", "--train_num", help="the training number which should be continued", required=False, default="None", type=str)
    parser.add_argument("-rn", "--ref_node", help="reference node in the graph", required=True, default=0, type=int) # 35 for neurips_small
    args = parser.parse_args()
    
    ENV_NAME = args.env_name

    PATH = pathlib.Path().resolve().parent
    BENCH_CONFIG_PATH = PATH / "configs" / (ENV_NAME + ".ini")
    DATA_PATH = PATH / "Datasets" / ENV_NAME / "DC" / "Benchmark4_medium"
    LOG_PATH = PATH / "lips_logs.log"
    
    # print(type(args.hidden_layers))
    # hidden_layers = eval(args.hidden_layers)
    print(f"Env used: {ENV_NAME}")
    print(f"Train mode: {args.train}")
    # print(f"hidden_layers: {hidden_layers}")
    print(f"Number of GNN layers: {args.num_gnn_layers}")
    print(f"Epochs: {args.epochs}")
    if args.load_path is not None:
        print(f"Load path: {args.load_path}")
    
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # or "cuda:0" if you have any GPU
    batch_size = args.batch_size
    
    benchmark = PowerGridBenchmark(dataset_path=DATA_PATH,
                                   benchmark_name="Benchmark4",
                                   load_data_set=True,
                                   config_path=BENCH_CONFIG_PATH,
                                   log_path=LOG_PATH)
    
    env, obs = get_obs(benchmark)
    n_bus = env.n_sub * 2
    sn_mva = env.backend._grid.get_sn_mva()
    
    if eval(args.train):
        train_loader, val_loader, test_loader, test_ood_loader = create_loaders(benchmark=benchmark, 
                                                                                batch_size=batch_size, 
                                                                                device=device)
    
    metrics_all = []
    # metrics_all["test"] = {}
    # metrics_all["test_ood"] = {}
    for train_num_ in range(args.n_train):
        if eval(args.train_num) is not None:
            train_num = eval(args.train_num)
        else:
            train_num = train_num_
        model = GPGmodel(ref_node=args.ref_node, sn_mva=sn_mva, num_gnn_layers=args.num_gnn_layers, device=device).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        loss_fn = nn.MSELoss()        
        
        if eval(args.train):
            print("Enter the training mode")
            model.train()
            
            model, train_losses, val_losses = train_model(model,
                                                          train_loader=train_loader,
                                                          optimizer=optimizer,
                                                          val_loader=val_loader,
                                                          loss_fn=loss_fn,
                                                          epochs=args.epochs,
                                                          scheduler=scheduler,
                                                          save_freq=50,
                                                          save_path=args.save_path,
                                                          sn_mva=sn_mva,
                                                          train_continue=eval(args.train_continue),
                                                          load_path=args.load_path,
                                                          model_name=args.model_name,
                                                          train_num=train_num                                                        
                                                          )
            
        else:
            print("Enter the evaluation mode")
            if (args.load_path is None) or not(os.path.exists(args.load_path)):
                raise FileNotFoundError("The load_path is not indicated or not existing")
            
            test_loader, test_ood_loader = create_loaders(benchmark=benchmark, 
                                                        batch_size=batch_size, 
                                                        device=device,
                                                        eval=True)
            
            checkpoint = torch.load(os.path.join(args.load_path, f"train_{train_num}", args.model_name), weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            train_losses = checkpoint["train_losses"]
            train_losses_data = checkpoint["train_losses_data"]
            train_losses_physics = checkpoint["train_losses_physics"]
            val_losses = checkpoint["val_losses"]
            
        metrics = {}
        metrics["test"] = evaluate(model, test_loader, benchmark,  dataset="_test_dataset", sn_mva=sn_mva)
        metrics["test_ood"] = evaluate(model, test_ood_loader, benchmark, dataset="_test_ood_topo_dataset", sn_mva=sn_mva)
        
        metrics_all.append(metrics)
        
        with open(os.path.join(args.save_path, f'result_{train_num}.json'), "w", encoding="utf-8") as f:
            json.dump(obj=metrics, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
    
    compute_mean_std(metrics_all, save_path=args.save_path)
        