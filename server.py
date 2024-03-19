from collections import OrderedDict

from omegaconf import DictConfig

import torch

from model import Predictor, test
from dataset import prepare_dataset

from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import pickle as pkl

import time
import datetime

history = {"server_eval":[], "eval_timestep":[], "client_loss": [], }

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = Predictor()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, mean_loss = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"mean_loss": mean_loss, "timestep": time.time()}

    return evaluate_fn

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    return {"metrics": metrics}

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.datasets, cfg.num_clients, cfg.batch_size, is_uniform=cfg.uniform_data_distribution
    )

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(testloader),
        evaluate_metrics_aggregation_fn=weighted_average,
    )  # a function to run on the server side to evaluate the global model.

    # Start Flower server
    history = fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    curtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"server_result_{curtime}.pkl", 'wb') as f:
        pkl.dump(history, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()