import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np

from src.model import RNNPredictor

def train(
    model: RNNPredictor,
    trainloader: DataLoader,
    validationloader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    # Config set up
    with open("config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    predictor_cfg = cfg["Predictor"]
    learning_rate = predictor_cfg["learning_rate"]

    """Train the network."""
    # Define loss and optimizer
    optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    model.to(device)
    
    for epoch in range(epochs):
        training_loss = []
        val_loss = []
        for step, (input_batch, target_batch) in enumerate(trainloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = model(input_batch, target_batch)
            cur_loss = criterion(outputs, target_batch)
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            training_loss.append(cur_loss.item())
        with torch.no_grad():
            model.eval()
            for _, (input_batch, target_batch) in enumerate(validationloader):            
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                outputs = model(input_batch, target_batch)
                cur_loss = criterion(outputs, target_batch)
                val_loss.append(cur_loss.item())
            model.train()
        print(f"Epoch: {epoch + 1}| Loss: {np.mean(training_loss):.6f}| Val Loss: {np.mean(val_loss):.6f}")