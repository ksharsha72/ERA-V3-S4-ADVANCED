from flask import Flask, render_template, request, jsonify, Response
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNNModel, train_model, test_model
import json
import torch.nn.functional as F

app = Flask(__name__)

# Global variables
OPTIMIZERS = {"adam": optim.Adam, "sgd": optim.SGD, "rmsprop": optim.RMSprop}
# Store models and their configurations
models = {}


def get_data_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4 if torch.cuda.is_available() else 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=True,
        pin_memory=True,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    return train_loader, test_loader


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    model_id = data["model_id"]

    # Get parameters from request
    layer_kernels = [int(k) for k in data["kernels"]]
    batch_size = int(data["batch_size"])
    optimizer_name = data["optimizer"]
    learning_rate = float(data["learning_rate"])
    num_epochs = int(data["epochs"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create or get model
    if model_id not in models:
        models[model_id] = CNNModel(layer_kernels).to(device)
    else:
        models[model_id] = CNNModel(layer_kernels).to(device)

    model = models[model_id]

    # Setup optimizer with custom learning rate
    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr=learning_rate)

    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size)

    def generate():
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            correct_train = 0
            total_train = 0
            batch_count = 0

            # Training phase
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                # Calculate training accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct_train += pred.eq(target.view_as(pred)).sum().item()
                total_train += target.size(0)

                epoch_loss += loss.item()
                batch_count += 1

                if batch_idx % 100 == 0:
                    yield json.dumps(
                        {
                            "model_id": model_id,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": loss.item(),
                        }
                    )

            # Calculate training accuracy
            train_accuracy = 100.0 * correct_train / total_train

            # Test phase
            test_loss, test_accuracy = test_model(model, device, test_loader)
            yield json.dumps(
                {
                    "model_id": model_id,
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "accuracy": test_accuracy,
                    "train_accuracy": train_accuracy,
                    "avg_train_loss": epoch_loss / batch_count,
                }
            )

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)
