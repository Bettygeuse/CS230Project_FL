import torch
import yaml

from src.model import RNNPredictor
from dataset import load_data
from train import train
from test import test

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Centralized PyTorch training")
    print("Load data")
    trainloader, validationloader, testloader, _ = load_data(1)

    print("Start training")
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    predictor_cfg = cfg["Predictor"]
    num_epochs = predictor_cfg["num_epochs"]

    hidden_size = predictor_cfg["hidden_size"]
    input_size = predictor_cfg["input_size"]
    output_size = predictor_cfg["output_size"]
    model = RNNPredictor(input_size=input_size
                    , output_size=output_size
                    , hidden_size=hidden_size
                    , use_cuda=(DEVICE == 'cuda'))
    train(model, trainloader=trainloader, validationloader = validationloader, epochs=num_epochs, device=DEVICE)

    print("Evaluate model")
    loss = test(model, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)

if __name__ == "__main__":
    main()