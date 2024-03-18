from typing import Tuple, Dict
import torch
import numpy as np
import pickle as pkl
import torch.utils.data as Data
from torch.utils.data import DataLoader
import yaml

import src.dataHelper as dh

def load_data(client_id: int) -> Tuple[torch.utils.data.DataLoader, 
                                       torch.utils.data.DataLoader, 
                                       torch.utils.data.DataLoader, Dict]:
    """Load CACC dataset"""
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    predictor_cfg = cfg["Predictor"]
    batch_size = predictor_cfg["batch_size"]
    window_size = predictor_cfg["window_size"]
    period = predictor_cfg["period"]
    training_dataset_path = cfg[f"traindata_path_client_{client_id}"]
    with open(training_dataset_path, 'rb') as fp:
        traj_dict = pkl.load(fp)

    ori_window_groups = dh.traj_dict2windows(traj_dict,  window_size, period)
    # convert positions of ego_vehicle and of leading vehicle to distance
    window_groups = []
    for windows in ori_window_groups:
        distance_windows = windows[..., [0,1,2,4,5,6]]
        distance_windows[..., 0] = windows[...,3] - windows[...,0]
        window_groups.append(distance_windows)

    split1, split2 = int(len(window_groups) * 0.7), int(len(window_groups) * 0.9)
    training_window_groups = window_groups[:split1]
    validation_window_groups = window_groups[split1:split2]
    test_window_groups = window_groups[split2:]

    training_windows = np.concatenate(training_window_groups, axis=0)
    validation_windows = np.concatenate(validation_window_groups, axis=0)
    test_windows = np.concatenate(test_window_groups, axis=0)

    min_val = np.min(training_windows.reshape(-1, training_windows.shape[-1]), axis=0)
    max_val = np.max(training_windows.reshape(-1, training_windows.shape[-1]), axis=0)

    training_windows = (training_windows-min_val)/(max_val-min_val)
    validation_windows = (validation_windows-min_val)/(max_val-min_val)
    test_windows = (test_windows-min_val)/(max_val-min_val)

    training_dataset = Data.TensorDataset(
        torch.tensor(training_windows[:,:-10,:]).type(torch.FloatTensor),
        torch.tensor(training_windows[:,-10:,:3]).type(torch.FloatTensor)
    )
    validation_dataset = Data.TensorDataset(
        torch.tensor(validation_windows[:,:-10,:]).type(torch.FloatTensor),
        torch.tensor(validation_windows[:,-10:,:3]).type(torch.FloatTensor)
    )
    test_dataset = Data.TensorDataset(
        torch.tensor(test_windows[:,:-10,:]).type(torch.FloatTensor),
        torch.tensor(test_windows[:,-10:,:3]).type(torch.FloatTensor)
    )
    training_loader = DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    num_examples = {"trainset" : training_windows.shape[0], 
                    "validationset": validation_windows.shape[0], 
                    "testset" : test_windows.shape[0]}
    
    return training_loader, validation_loader, test_loader, num_examples