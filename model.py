import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import torch.cuda as cuda


class CNNModel(nn.Module):
    def __init__(self, layer_kernels):
        super(CNNModel, self).__init__()
        # Define the layers with configurable kernels
        self.conv1 = nn.Conv2d(1, layer_kernels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            layer_kernels[0], layer_kernels[1], kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            layer_kernels[1], layer_kernels[2], kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            layer_kernels[2], layer_kernels[3], kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(layer_kernels[3] * 1 * 1, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = x.view(-1, self.conv4.out_channels * 1 * 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def to_device(self, device):
        """Helper method to move model to specified device"""
        self.to(device)
        return self


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    # Move to CUDA if available
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Clear GPU cache before training

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU in non-blocking mode for better performance
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            # Convert tensor to float before JSON serialization
            loss_value = loss.item()
            log_data = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss_value,
                "progress": f"{batch_idx * len(data)}/{len(train_loader.dataset)}",
                "device": str(device),
            }
            print(json.dumps(log_data))
            yield json.dumps(log_data)

        # Optional: Clear cache periodically for long training sessions
        if batch_idx % 500 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy
