import pickle as pkl
import numpy as np
import dataHelper as dh

WINDOW_SIZE = 50
PERIOD = 10

def preprocess_dataset(dataset):
    with open(dataset, 'rb') as fp:
        traj_dict = pkl.load(fp)
    ori_window_groups = dh.traj_dict2windows(traj_dict, WINDOW_SIZE, PERIOD)
    
    # convert positions of ego_vehicle and of leading vehicle to distance
    window_groups = []
    for windows in ori_window_groups:
        distance_windows = windows[..., [0,1,2,4,5,6]]
        distance_windows[..., 0] = windows[...,3] - windows[...,0]
        window_groups.append(distance_windows)

    windows = np.concatenate(window_groups, axis=0)
    min_val = np.min(windows.reshape(-1, windows.shape[-1]), axis=0)
    max_val = np.max(windows.reshape(-1, windows.shape[-1]), axis=0)

    windows = (windows - min_val) / (max_val - min_val)
    np.save(dataset.replace(".pkl", ".npy"), windows)

datasets = ["type1_data.pkl", "type2_data.pkl", "type3_data.pkl"]

for dataset in datasets:
    preprocess_dataset(dataset)