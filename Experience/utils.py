import os
import json
import numpy 
import copy
import torch
import math
import numpy as np
from torch_geometric.nn import MessagePassing
from pprint import pprint
from lips.metrics.ml_metrics.external_metrics import mape_quantile

class NpEncoder(json.JSONEncoder):
    """
    taken from : https://java2blog.com/object-of-type-int64-is-not-json-serializable/
    """
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        # if the object is a function, save it as a string
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)


class LocalConservationLayer(MessagePassing):
    """Compute local conservation error

    This class computes the local conservation error without any update of voltage angles.

    Args:
        MessagePassing (_type_): _description_
    """
    def __init__(self, sn_mva):
        super().__init__(aggr="add")
        self.thetas = None
        self.sn_mva = sn_mva
        
    def forward(self, batch, thetas=None):
        # theta from previous GNN layer
        self.thetas = thetas

        # The difference with GPG layers resides also in propagation which gets the edge_index
        # with self loops (with diagonal elements of adjacency matrix)
        aggr_message = self.propagate(batch.edge_index,
                                      y=self.thetas,
                                      edge_weights=batch.edge_attr * self.sn_mva)

        input_node_power = (batch.x[:,0] - batch.x[:,1]).view(-1,1)
        # compute the local conservation error (at node level)
        nodal_error = input_node_power - aggr_message

        return nodal_error, input_node_power, aggr_message

    def message(self, y_i, y_j, edge_weights):
        """
        Compute the message
        """
        tmp = y_j * edge_weights.view(-1,1)
        return tmp
    
    
def compute_local_conservation_error(test_loader, predictions_list: list, sn_mva: float, metrics: dict, obs):
    metrics["Physics"]["CHECK_LC"] = {}
    lc_layer= LocalConservationLayer(sn_mva=sn_mva)
    nodal_errors = []
    input_node_powers = []
    aggr_msgs = []
    
    for batch, pred in zip(test_loader, predictions_list):
        nodal_error, input_node_power, aggr_msg = lc_layer(batch, thetas=pred)
        nodal_errors.append(nodal_error)
        input_node_powers.append(input_node_power)
        aggr_msgs.append(aggr_msg)
        
    nodal_errors = torch.vstack(nodal_errors)
    input_node_powers = torch.vstack(input_node_powers).detach().cpu().numpy()
    aggr_msgs = torch.vstack(aggr_msgs).detach().cpu().numpy()
    
    nodal_errors_per_obs = nodal_errors.view(-1, obs.n_sub*2)
    array = nodal_errors_per_obs.detach().cpu().numpy()
    metrics["Physics"]["CHECK_LC"]["violation_percentage"] = np.round(np.sum(np.abs(array) > 0.1) / array.size, 4)*100
    
    # metrics["Physics"]["CHECK_LC"]["mape"] = np.mean(np.abs(input_node_powers - aggr_msgs)) / np.mean(np.abs(input_node_powers))
    metrics["Physics"]["CHECK_LC"]["mape"] = mape_quantile(input_node_powers, aggr_msgs, quantile=0.99)
    
    return metrics 

def compute_local_conservation_error_mlp(test_loader, predictions_list: list, sn_mva: float, metrics: dict, obs):
    metrics["Physics"]["CHECK_LC"] = {}
    lc_layer= LocalConservationLayer(sn_mva=sn_mva)
    nodal_errors = []
    input_node_powers = []
    aggr_msgs = []
    
    for batch_, pred in zip(test_loader, predictions_list):
        features, targets, batch = batch_
        nodal_error, input_node_power, aggr_msg = lc_layer(batch, thetas=pred.view(-1,1) * (math.pi/180))
        nodal_errors.append(nodal_error)
        input_node_powers.append(input_node_power)
        aggr_msgs.append(aggr_msg)
        
    nodal_errors = torch.vstack(nodal_errors)
    input_node_powers = torch.vstack(input_node_powers).detach().cpu().numpy()
    aggr_msgs = torch.vstack(aggr_msgs).detach().cpu().numpy()
    
    nodal_errors_per_obs = nodal_errors.view(-1, obs.n_sub*2)
    array = nodal_errors_per_obs.detach().cpu().numpy()
    metrics["Physics"]["CHECK_LC"]["violation_percentage"] = np.round(np.sum(np.abs(array) > 0.1) / array.size, 4)*100
    
    # metrics["Physics"]["CHECK_LC"]["mape"] = np.mean(np.abs(input_node_powers - aggr_msgs)) / np.mean(np.abs(input_node_powers))
    metrics["Physics"]["CHECK_LC"]["mape"] = mape_quantile(input_node_powers, aggr_msgs, quantile=0.99)
    
    return metrics 

def compute_mean_std(metrics, save_path):
    metrics_all = {}
    metrics_all["test"] = {}
    metrics_all["test"]["MAE"] = []
    metrics_all["test"]["MAPE90"] = []
    metrics_all["test"]["LC_violation"] = []
    metrics_all["test"]["LC_MAPE"] = []
    metrics_all["test"]["INF_TIME"] = []
    metrics_all["test"]["theta"] = {}
    metrics_all["test"]["theta"]["MAPE90"] = []
    metrics_all["test"]["theta"]["MAPE10"] = []
    metrics_all["test"]["theta"]["MAE"] = []
    metrics_all["test"]["theta"]["WMAPE"] = []
    
    metrics_all["test_ood"] = {}
    metrics_all["test_ood"]["MAE"] = []
    metrics_all["test_ood"]["MAPE90"] = []
    metrics_all["test_ood"]["LC_violation"] = []
    metrics_all["test_ood"]["LC_MAPE"] = []
    metrics_all["test_ood"]["INF_TIME"] = []
    metrics_all["test_ood"]["theta"] = {}
    metrics_all["test_ood"]["theta"]["MAPE90"] = []
    metrics_all["test_ood"]["theta"]["MAPE10"] = []
    metrics_all["test_ood"]["theta"]["MAE"] = []
    metrics_all["test_ood"]["theta"]["WMAPE"] = []
    
    mean_std = copy.copy(metrics_all)
    
    for metric in metrics:
        for dataset in ["test", "test_ood"]:
            metrics_all[dataset]["MAE"].append(metric[dataset]["ML"]["MAE_avg"]["p_or"])
            metrics_all[dataset]["MAPE90"].append(metric[dataset]["ML"]["MAPE_90_avg"]["p_or"])
            metrics_all[dataset]["LC_violation"].append(metric[dataset]["Physics"]["CHECK_LC"]["violation_percentage"])
            metrics_all[dataset]["LC_MAPE"].append(metric[dataset]["Physics"]["CHECK_LC"]["mape"])
            metrics_all[dataset]["INF_TIME"].append(metric[dataset]["INF_TIME"])
            metrics_all[dataset]["theta"]["MAPE10"].append(metric[dataset]["theta"]["MAPE10"])
            metrics_all[dataset]["theta"]["MAPE90"].append(metric[dataset]["theta"]["MAPE90"])
            metrics_all[dataset]["theta"]["MAE"].append(metric[dataset]["theta"]["MAE"])
            metrics_all[dataset]["theta"]["WMAPE"].append(metric[dataset]["theta"]["WMAPE"])
    
    for dataset in ["test", "test_ood"]:
        mean_std[dataset]["MAE"] = f"{np.mean(metrics_all[dataset]['MAE'])} +/- {np.std(metrics_all[dataset]['MAE'])}"
        mean_std[dataset]["MAPE90"] = f"{np.mean(metrics_all[dataset]['MAPE90'])} +/- {np.std(metrics_all[dataset]['MAPE90'])}"
        mean_std[dataset]["LC_violation"] = f"{np.mean(metrics_all[dataset]['LC_violation'])} +/- {np.std(metrics_all[dataset]['LC_violation'])}"
        mean_std[dataset]["LC_MAPE"] = f"{np.mean(metrics_all[dataset]['LC_MAPE'])} +/- {np.std(metrics_all[dataset]['LC_MAPE'])}"
        mean_std[dataset]["INF_TIME"] = f"{np.mean(metrics_all[dataset]['INF_TIME'])} +/- {np.std(metrics_all[dataset]['INF_TIME'])}"
        mean_std[dataset]["theta"]["MAPE10"] = f"{np.mean(metrics_all[dataset]['theta']['MAPE10'])} +/- {np.std(metrics_all[dataset]['theta']['MAPE10'])}"
        mean_std[dataset]["theta"]["MAPE90"] = f"{np.mean(metrics_all[dataset]['theta']['MAPE90'])} +/- {np.std(metrics_all[dataset]['theta']['MAPE90'])}"
        mean_std[dataset]["theta"]["MAE"] = f"{np.mean(metrics_all[dataset]['theta']['MAE'])} +/- {np.std(metrics_all[dataset]['theta']['MAE'])}"
        mean_std[dataset]["theta"]["WMAPE"] = f"{np.mean(metrics_all[dataset]['theta']['WMAPE'])} +/- {np.std(metrics_all[dataset]['theta']['WMAPE'])}"
        
    with open(os.path.join(save_path, f'final_results.json'), "w", encoding="utf-8") as f:
            json.dump(obj=mean_std, fp=f, indent=4, sort_keys=True, cls=NpEncoder)
            
    pprint(mean_std)
    