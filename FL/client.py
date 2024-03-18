from collections import OrderedDict
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

import numpy as np
import torch
import flwr as fl
import yaml

from src.model import RNNPredictor
from dataset import load_data
from train import train
from test import test


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class centralizedClient(fl.client.NumPyClient):
    """Flower client implementing centralized-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: RNNPredictor,
        trainloader: DataLoader,
        validationloader: DataLoader,
        testloader: DataLoader,
        num_examples: Dict,
        num_epoches: int
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.num_epoches = num_epoches

    def get_parameters(self, config) -> List[np.ndarray]:
        print("get param is called")
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        print("set param is called")
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        print("fit is called")
        self.set_parameters(parameters)
        train(self.model, self.trainloader, self.validationloader, epochs=self.num_epoches, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        print("evaluates is called")
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
def main() -> None:
    """Load data, start centralizedClient."""

    # Load model and data
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

    trainloader, validationloader, testloader, num_examples = load_data(1)

    # Start client
    fl.client.start_numpy_client(server_address="172.31.0.44:8080", client=centralizedClient(model, trainloader, validationloader, testloader, num_examples, num_epochs))


if __name__ == "__main__":
    main()