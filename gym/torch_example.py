import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


class FFNN(nn.Module):
    """Simple Feed Forward Neural Network with n hidden layers"""

    def __init__(
        self, input_size, num_hidden_layers, hidden_size, out_size, accuracy_function
    ):
        super().__init__()
        self.accuracy_function = accuracy_function

        # Create first hidden layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Create remaining hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, out_size)

    def forward(self, input_image):
        # Flatten image
        input_image = input_image.view(input_image.size(0), -1)

        # Utilize hidden layers and apply activation function
        output = self.input_layer(input_image)
        output = F.relu(output)

        for layer in self.hidden_layers:
            output = layer(output)
            output = F.relu(output)

        # Get predictions
        output = self.output_layer(output)
        return output

    def training_step(self, batch):
        # Load batch
        images, labels = batch

        # Generate predictions
        output = self(images)

        # Calculate loss
        loss = F.cross_entropy(output, labels)
        return loss

    def validation_step(self, batch):
        # Load batch
        images, labels = batch

        # Generate predictions
        output = self(images)

        # Calculate loss
        loss = F.cross_entropy(output, labels)

        # Calculate accuracy
        acc = self.accuracy_function(output, labels)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]

        # Combine losses and return mean value
        epoch_loss = torch.stack(batch_losses).mean()

        # Combine accuracies and return mean value
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch: {} - Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
                epoch, result["val_loss"], result["val_acc"]
            )
        )


class ModelTrainer:
    def fit(
        self,
        epochs,
        learning_rate,
        model,
        train_loader,
        val_loader,
        opt_func=torch.optim.SGD,
    ):
        history = []
        optimizer = opt_func(model.parameters(), learning_rate)

        for epoch in range(epochs):
            for batch in train_loader:
                loss = model.training_step(self._to_cuda(batch))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = self._evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

        return history

    def _evaluate(self, model, val_loader):
        outputs = [model.validation_step(self._to_cuda(batch)) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def _to_cuda(self, batch):
        return (batch[0].to("cuda"), batch[1].to("cuda"))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def plot_history(history):
    losses = [x["val_loss"] for x in history]
    plt.plot(losses, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Loss and Accuracy")


if __name__ == "__main__":
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")

    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    data = FashionMNIST(root="data/", download=True, transform=ToTensor())

    validation_size = int(0.2 * len(data))
    train_size = len(data) - validation_size

    print(f"Number of training samples: {train_size}")
    print(f"Number of training samples: {validation_size}")

    train_data, val_data = random_split(data, [train_size, validation_size])

    batch_size = 128

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)
    model = FFNN(
        input_size=784,
        num_hidden_layers=3,
        hidden_size=32,
        out_size=10,
        accuracy_function=accuracy,
    ).to("cuda")
    print(model)

    model_trainer = ModelTrainer()

    training_history = []
    training_history += model_trainer.fit(
        epochs=10,
        learning_rate=0.2,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # plot_history(training_history)
