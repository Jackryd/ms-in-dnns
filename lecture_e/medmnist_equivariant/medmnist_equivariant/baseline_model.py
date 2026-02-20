"""
Baseline (non-equivariant) CNN model for MedMNIST.

TODO: Implement the BaselineCNN and PLBaselineModule classes.
"""
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy


class BaselineCNN(nn.Module):
    """
    Standard CNN baseline for MedMNIST classification.

    TODO: Implement a simple CNN with 3 convolutional blocks:
    - Each block: Conv2d -> ReLU -> MaxPool2d
    - Followed by a classifier (Linear layers)
    - Input: 28x28 images with `in_channels` channels
    - Output: `num_classes` logits
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # TODO: Implement the model architecture
        # Suggested structure:
        # - Block 1: in_channels -> 32 channels
        # - Block 2: 32 -> 64 channels
        # - Block 3: 64 -> 128 channels
        # - Classifier: flatten -> Linear -> ReLU -> Linear
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    # differs from the cifar10_simple model, but
    # aligns more with the description above
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PLBaselineModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for BaselineCNN.

    TODO: Implement training_step, validation_step, test_step, and configure_optimizers.
    """

    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        # TODO: Initialize model, loss function, and metrics
        self.model = BaselineCNN(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward pass through the model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # TODO: Implement training step
        # - Compute loss and accuracy
        # - Log metrics with self.log()
        inputs, labels = batch
        if labels.ndim > 1:
            labels = labels.squeeze(1)
        labels = labels.long()

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: Implement validation step
        inputs, labels = batch
        if labels.ndim > 1:
            labels = labels.squeeze(1)
        labels = labels.long()

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # TODO: Implement test step
        inputs, labels = batch
        if labels.ndim > 1:
            labels = labels.squeeze(1)
        labels = labels.long()

        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)

        self.log("best/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("best/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # TODO: Return optimizer (e.g., Adam)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
