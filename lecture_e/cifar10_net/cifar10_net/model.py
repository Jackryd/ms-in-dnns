import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
import wandb

class SimpleCIFARNet(nn.Module):
    """
    Simplified VGG-style architecture for CIFAR10.

    This is a fixed architecture with dropout regularization.
    It is provided pre-trained for use in the adversarial attacks exercise.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            # Block 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            # Block 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_pretrained(checkpoint_path: str, device: str = "cpu") -> SimpleCIFARNet:
    """Load a pre-trained SimpleCIFARNet from a checkpoint file."""
    model = SimpleCIFARNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

class PLCIFARModule(L.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleCIFARNet(num_classes, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_classes = num_classes

        metrics = torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.best_metrics = metrics.clone(prefix="best/")
        self.test_conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        self.train_metrics.update(preds, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        self.val_metrics.update(preds, labels)
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)
        self.log("step", float(self.current_epoch + 1), on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)

        self.best_metrics.update(preds, labels)
        self.test_conf_mat.update(preds, labels)
        self.log("best/loss", loss, on_epoch=True, on_step=False)
        self.log_dict(self.best_metrics, on_epoch=True, on_step=False)

    def on_test_epoch_end(self):
        counts = self.test_conf_mat.compute().detach().cpu()
        class_names = self.trainer.datamodule.CLASS_NAMES
        data = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                data.append([class_names[i], class_names[j], int(counts[i, j].item())])
        fields = {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"}
        conf_mat = wandb.plot_table(
            "wandb/confusion_matrix/v1",
            wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=data),
            fields,
            {"title": "confusion matrix on best epoch"},
            split_table=True,
        )
        wandb.log({"best/conf_mat": conf_mat})
        self.test_conf_mat.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
