from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch

import centralized
import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class centralizedClient(fl.client.NumPyClient):
    """Flower client implementing centralized-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: centralized.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

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
        centralized.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        print("evaluates is called")
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = centralized.test(self.model, self.testloader, device=DEVICE)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}
    
def main() -> None:
    """Load data, start centralizedClient."""

    # Load model and data
    model = centralized.Net()
    model.to(DEVICE)
    trainloader, testloader, num_examples = centralized.load_data()

    # Start client
    fl.client.start_numpy_client(server_address="172.31.0.44:8080", client=centralizedClient(model, trainloader, testloader, num_examples))


if __name__ == "__main__":
    main()