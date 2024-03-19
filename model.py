
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Note the net and functions here defined do not have any FL-specific components.
HIDDEN_SIZE = 120
INPUT_SIZE = 6
OUTPUT_SIZE = 3

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.rnn = nn.GRU(
             input_size=INPUT_SIZE,
             hidden_size=HIDDEN_SIZE,
             num_layers=2,
             dropout=0.15,
             batch_first=True, 
             bidirectional=True
        )
        self.out = nn.Linear(HIDDEN_SIZE * 2 * 40, OUTPUT_SIZE * 10)
        self.tanh = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        rnn_out, _ = self.rnn(inputs, None)
        #print(rnn_out.size())
        fc_out = self.out(rnn_out.contiguous().view(batch_size, -1))
        outputs = self.tanh(fc_out)
        outputs = outputs.view(-1, 10, OUTPUT_SIZE)
        return outputs


def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.MSELoss()
    net.train()
    net.to(device)
    for epoch in range(epochs):
        training_loss = []
        for step, (input_batch, target_batch) in enumerate(trainloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = net(input_batch)
            cur_loss = criterion(outputs, target_batch)
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            training_loss.append(cur_loss.item())
        print(f"Epoch: {epoch + 1}| Loss: {np.mean(training_loss):.6f}")


def test(net: Predictor, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.MSELoss()
    loss = 0
    losses = []
    net.eval()
    net.to(device)
    with torch.no_grad():
        for _, (input_batch, target_batch) in enumerate(testloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            outputs = net(input_batch)
            cur_loss = criterion(outputs, target_batch).item()
            loss += cur_loss
            losses.append(cur_loss)
            # TODO: probably mean loss makes more sense
    return loss, np.mean(losses)
