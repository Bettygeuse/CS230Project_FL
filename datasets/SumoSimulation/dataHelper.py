import numpy as np

def step_list2np_array(traj_list):
    '''
    convert a trajectory from list to numpy array 
    '''
    traj_series = []
    for step_list in traj_list:

        step_vector = np.zeros(7)
        step_vector[:6] = np.array(step_list[:6])
        if step_list[6] != None:
            step_vector[6] = 1
        traj_series.append(step_vector)
    return np.array(traj_series)

def traj2windows(traj_series, window_size, stride):
    '''
     split one trajectory to windows
    '''
    windows = []
    for i in range(0, len(traj_series), stride):
        if i + window_size > len(traj_series):
            # Notice that the last few steps may be fewer then window size
            # We discard these steps in this case
            break
        windows.append(traj_series[i:i+window_size])
    return windows

def traj_dict2windows(traj_dict, window_size, stride):
    '''
        vector of trajectory in each step:
            posX, speed, acceleration, preceding_posX, preceding_speed, preceding_acceleration, preceding_vid)
    '''
    windows_groups = []
    for traj in traj_dict.values():
        traj_list = traj["steps"]
        traj_series = step_list2np_array(traj_list)
        # splitting sliding windows
        windows = traj2windows(traj_series, window_size, stride)
        if len(windows) > 0:
            windows = np.array(windows).astype(float)
            windows_groups.append(windows)
    return windows_groups
