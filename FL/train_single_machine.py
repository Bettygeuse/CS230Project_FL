import yaml
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import argparse
import torch
from torch.optim import Adam
import torch.utils.data as Data
from torch.utils.data import DataLoader

import src.dataHelper as dh
from src.model import RNNPredictor

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='config_path', default="config.yaml")
args = parser.parse_args()

with open(args.config_path, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

predictor_cfg = cfg["Predictor"]
num_epochs = predictor_cfg["num_epochs"]
batch_size = predictor_cfg["batch_size"]
learning_rate = predictor_cfg["learning_rate"]

window_size = predictor_cfg["window_size"]
period = predictor_cfg["period"]
training_dataset_path = cfg["traindata_path"]

with open(training_dataset_path, 'rb') as fp:
    traj_dict = pkl.load(fp)

ori_window_groups = dh.traj_dict2windows(traj_dict,  window_size, period)
# convert positions of ego_vehicle and of leading vehicle to distance
window_groups = []
for windows in ori_window_groups:
    distance_windows = windows[..., [0,1,2,4,5,6]]
    distance_windows[..., 0] = windows[...,3] - windows[...,0]
    window_groups.append(distance_windows)

training_window_groups = window_groups[:-len(window_groups)//4]
validation_window_groups = window_groups[-len(window_groups)//4:]

training_windows = np.concatenate(training_window_groups, axis=0)
validation_windows = np.concatenate(validation_window_groups, axis=0)

min_val = np.min(training_windows.reshape(-1, training_windows.shape[-1]), axis=0)
max_val = np.max(training_windows.reshape(-1, training_windows.shape[-1]), axis=0)

np.save(predictor_cfg["min_path"], min_val)
np.save(predictor_cfg["max_path"], max_val)

training_windows = (training_windows - min_val) / (max_val-min_val)
validation_windows = (validation_windows - min_val) / (max_val-min_val)

training_dataset = Data.TensorDataset(
    torch.tensor(training_windows[:,:-10,:]).type(torch.FloatTensor),
    torch.tensor(training_windows[:,-10:,:3]).type(torch.FloatTensor)
)
validation_dataset = Data.TensorDataset(
    torch.tensor(validation_windows[:,:-10,:]).type(torch.FloatTensor),
    torch.tensor(validation_windows[:,-10:,:3]).type(torch.FloatTensor)
)
training_loader = DataLoader(
    dataset=training_dataset, batch_size=batch_size, shuffle=True
)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=batch_size, shuffle=True
)

hidden_size = predictor_cfg["hidden_size"]
input_size = predictor_cfg["input_size"]
output_size = predictor_cfg["output_size"]

model = RNNPredictor(input_size=input_size
                    , output_size=output_size
                    , hidden_size=hidden_size
                    , use_cuda=True)

model.cuda()

optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = torch.nn.MSELoss()
min_val_loss = 0.006

for epoch in range(num_epochs):
    training_loss = []
    val_loss = []
    for input_batch, target_batch in training_loader:        
        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()
        outputs = model(input_batch, target_batch)
        cur_loss = criterion(outputs, target_batch)
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        training_loss.append(cur_loss.item())
    with torch.no_grad():
        model.eval()
        for input_batch, target_batch in validation_loader:            
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
            outputs = model(input_batch, target_batch)
            cur_loss = criterion(outputs, target_batch)
            val_loss.append(cur_loss.item())
        model.train()
    print(f"Epoch: {epoch + 1}| Loss: {np.mean(training_loss):.6f}| Val Loss: {np.mean(val_loss):.6f}")
    
    if np.mean(val_loss) < min_val_loss:
        model_name = f"predictorDVA_params_{np.mean(val_loss)}.pkl"
        torch.save(model.state_dict(), f"model/{model_name}")
        
        min_val_loss = np.mean(val_loss)
    if epoch % 100 == 0 and epoch != 0:
        model_name = f"predictorDVA_params_{np.mean(val_loss)}.pkl"
        torch.save(model.state_dict(), f"model/{model_name}")

else:
    if num_epochs > 0:
        model_name = f"predictorDVA_params_final_{np.mean(val_loss)}.pkl"
        torch.save(model.state_dict(), f"model/{model_name}")

with torch.no_grad():
    errors = []
    model.eval()
    for sample_step, window_group in enumerate(validation_window_groups):
        normalized_windows = (window_group - min_val) / (max_val-min_val)
        window_group_input = torch.tensor(normalized_windows).type(torch.FloatTensor).cuda()
        
        output = model(window_group_input[:, :-10, :], window_group_input[:, :, :3]).cpu().numpy()
        prd_windows = output * (max_val[:3]-min_val[:3]) + min_val[:3]
        target = np.concatenate(window_group[:, :period, :], axis=0)
        target = np.concatenate((target, window_group[-1, period:, :]), axis=0)
        target = target[40:] # length * dims to match the reconstructed
        assert period <= 10 # make sure the mapping is the same
        predict = np.concatenate(prd_windows[:, :period, :], axis=0)
        predict = np.concatenate((predict, prd_windows[-1, period:, :]), axis=0)
        
        errors.append(target[:, :3] - predict)
        plt.suptitle("reconstruct sample #%d"%sample_step)


errors = np.absolute( np.concatenate(errors, axis=0) )
error_mean = np.mean(errors, axis=0)
error_cov = np.cov(errors.T)
np.save("model/error_mean_%s.npy"%model_name.replace("_params", "").replace(".pkl", ""), error_mean)
np.save("model/error_cov_%s.npy"%model_name.replace("_params", "").replace(".pkl", ""), error_cov)