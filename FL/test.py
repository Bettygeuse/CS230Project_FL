import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from typing import Tuple

from src.model import RNNPredictor

def test(
    model: RNNPredictor,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Test the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    test_loss = []
    with torch.no_grad():
        for input_batch, target_batch in testloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model(input_batch, target_batch)
            cur_loss = criterion(outputs, target_batch)
            test_loss.append(cur_loss.item())
    return np.mean(test_loss)