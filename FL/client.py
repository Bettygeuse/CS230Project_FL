import yaml
import torch
from src.model import RNNPredictor
from dataset import load_data
from train import train

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

predictor_cfg = cfg["Predictor"]
num_epochs = predictor_cfg["num_epochs"]
batch_size = predictor_cfg["batch_size"]
learning_rate = predictor_cfg["learning_rate"]

window_size = predictor_cfg["window_size"]
period = predictor_cfg["period"]
training_dataset_path = cfg["traindata_path"]

hidden_size = predictor_cfg["hidden_size"]
input_size = predictor_cfg["input_size"]
output_size = predictor_cfg["output_size"]

model = RNNPredictor(input_size=input_size
                    , output_size=output_size
                    , hidden_size=hidden_size
                    , use_cuda=(DEVICE == 'cuda'))

training_loader, validation_loader, test_loader, num_examples = load_data(1)
train(model, training_loader, validation_loader, 40, DEVICE)
