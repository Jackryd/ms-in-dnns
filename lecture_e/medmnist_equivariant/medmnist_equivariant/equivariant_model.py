"""
C4-Equivariant CNN model for MedMNIST using escnn.

TODO: Implement the C4EquivariantCNN and PLC4EquivariantModule classes.
"""
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy

# escnn imports
from escnn import gspaces
from escnn import nn as enn


class C4EquivariantCNN(nn.Module):
    """
    C4-equivariant CNN using escnn library.

    The C4 group consists of rotations by 0, 90, 180, and 270 degrees.
    This model maintains equivariance to these rotations throughout the
    feature extraction layers, then uses GroupPooling to produce
    rotation-invariant features for classification.

    TODO: Implement the equivariant architecture using escnn.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        # TODO: Define the symmetry group
        grp = gspaces.rot2dOnR2(N=4)
        # TODO: Define input field type
        # The input is a scalar field (trivial representation)
        in_type = enn.FieldType(grp, in_channels * [grp.trivial_repr]) 
        self.input_type = in_type

        # Tuned to match baseline parameter count as closely as possible.
        t1 = enn.FieldType(grp, 28 * [grp.regular_repr])
        t2 = enn.FieldType(grp, 56 * [grp.regular_repr])
        t3 = enn.FieldType(grp, 108 * [grp.regular_repr])

        # TODO: Build equivariant feature extractor
        # Use enn.R2Conv for equivariant convolutions
        # Use enn.ReLU for equivariant nonlinearity
        # Use enn.PointwiseMaxPool for equivariant pooling
        # Use enn.InnerBatchNorm for equivariant batch normalization (optional)
        # docs: https://quva-lab.github.io/escnn/api/escnn.nn.html?highlight=conv2d
        self.features = enn.SequentialModule(
            enn.R2Conv(in_type, t1, 3, 1),
            enn.InnerBatchNorm(t1),
            enn.ReLU(t1, inplace=True),
            enn.PointwiseMaxPool(t1, kernel_size=2, stride=2),

            enn.R2Conv(t1, t2, kernel_size=3, padding=1),
            enn.InnerBatchNorm(t2),
            enn.ReLU(t2, inplace=True),
            enn.PointwiseMaxPool(t2, kernel_size=2, stride=2),

            enn.R2Conv(t2, t3, kernel_size=3, padding=1),
            enn.InnerBatchNorm(t3),
            enn.ReLU(t3, inplace=True),
            enn.PointwiseMaxPool(t3, kernel_size=2, stride=2),
        )

        # TODO: Use enn.GroupPooling to convert equivariant features to invariant features
        # This pools over the group dimension, producing rotation-invariant features
        self.pool = enn.GroupPooling(t3)

        # TODO: Standard classifier on invariant features
        # self.classifier = nn.Sequential(...)
        self.classifier = nn.Sequential(
            nn.Linear(108 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        # 1. Wrap input as GeometricTensor: x = enn.GeometricTensor(x, self.input_type)
        # 2. Pass through equivariant layers
        # 3. Apply group pooling to get invariant features
        # 4. Extract tensor and flatten: x = x.tensor; x = torch.flatten(x, 1)
        # 5. Pass through classifier
        x = enn.GeometricTensor(x, self.input_type)
        x = self.features(x)
        x = self.pool(x)
        x = x.tensor; x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PLC4EquivariantModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for C4EquivariantCNN.

    TODO: Implement training_step, validation_step, test_step, and configure_optimizers.
    This should be very similar to PLBaselineModule.
    """

    def __init__(self, in_channels: int, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        # TODO: Initialize model, loss function, and metrics
        self.model = C4EquivariantCNN(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
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
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
