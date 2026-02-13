# See report here: https://wandb.ai/jackryd-chalmers-university-of-technology/ms-in-dnns-income-net/reports/Assignment-D--VmlldzoxNTkyNzEzNA
import argparse
import copy
from datetime import datetime
import os
import sys
import json
import pathlib as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import wandb

if "LOG_PATH" in os.environ:
    os.makedirs(os.path.dirname(os.environ["LOG_PATH"]), exist_ok=True)
    log = open(os.environ["LOG_PATH"], "a")
    sys.stdout = log
    sys.stderr = log


class IncomeNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AdultDataset(Dataset):
    """Adult UCI dataset, download data from https://archive.ics.uci.edu/dataset/2/adult"""

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        # one-hot encoding of categorical variables (including label)
        df = pd.get_dummies(df).astype("int32")

        data_columns = df.columns[:-2]
        labels_column = df.columns[-2:]
        self.data = torch.tensor(df[data_columns].values, dtype=torch.float32)
        self.labels = torch.tensor(df[labels_column].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ResampledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        class_to_indices = {}
        for i in range(len(dataset)):
            _, y = dataset[i]
            c = int(torch.argmax(y).item())
            class_to_indices.setdefault(c, []).append(i)

        max_count = max(len(v) for v in class_to_indices.values())
        indices = []
        for cls_indices in class_to_indices.values():
            indices.extend(cls_indices)
            need = max_count - len(cls_indices)
            if need > 0:
                full_repeats, remainder = divmod(need, len(cls_indices))
                indices.extend(cls_indices * full_repeats)
                if remainder > 0:
                    pick = torch.randperm(len(cls_indices))[:remainder].tolist()
                    indices.extend([cls_indices[j] for j in pick])

        perm = torch.randperm(len(indices))
        self.indices = [indices[i] for i in perm.tolist()]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_wandb_key():
    json_file = pl.Path("..", "wandb_key.json")
    if json_file.is_file():
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def main(args):

    wandb.login(key=get_wandb_key())
    wandb.init(project="ms-in-dnns-income-net", config=args, name=args.run_name)

    torch.manual_seed(0xDEADBEEF)

    if "LOG_PATH" in os.environ:
        bucket_name = os.environ["BUCKET"].split("gs://")[1]
        data_file = pl.PurePosixPath("/gcs", bucket_name, "adult_data", "adult.data")
        checkpoint_dir = pl.Path(os.path.dirname(os.environ["LOG_PATH"]))
    else:
        data_file = pl.PurePath("..", "data", "adult_data", "adult.data")
        checkpoint_dir = pl.Path(".")

    entire_dataset = AdultDataset(str(data_file))
    train_dataset, val_dataset = random_split(
        entire_dataset, [args.train_share, 1 - args.train_share]
    )
    if args.resampled_dataset:
        train_dataset = ResampledDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = IncomeNet(train_dataset[0][0].shape[0], train_dataset[0][1].shape[0])
    model = model.to(device)

    if args.weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        loss_weight = torch.tensor([args.weight, 1 - args.weight], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    last_ckpt_path = checkpoint_dir / "last_checkpoint.pt"
    best_val_loss_ckpt_path = checkpoint_dir / "lowest_val_loss_checkpoint.pt"
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_acc_epoch = -1
    best_acc_model_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.cpu().detach().item()
        train_loss = total_loss / len(train_loader)
        scheduler.step()
        model.eval()
        total_loss = 0.0
        true_pos = 0
        y_true_all = []
        pred_all = []
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.cpu().item()
            preds = torch.argmax(outputs, dim=-1)
            y_true_all.extend(torch.argmax(labels, dim=-1).cpu().tolist())
            pred_all.extend(preds.cpu().tolist())
            true_pos += int((preds == torch.argmax(labels, dim=-1)).cpu().sum().item())
        val_loss = total_loss / len(val_loader)
        val_acc = true_pos / len(val_dataset)
        epoch_step = epoch + 1
        print(
            f"Epoch [{epoch_step}/{args.epochs}]",
            f"Train Loss: {train_loss:.4f}",
            f"Val Loss: {val_loss:.4f}",
            f"Val Acc: {val_acc:.4f}",
        )

        wandb.log({"loss": {"train": train_loss, "val": val_loss}}, step=epoch_step)
        wandb.log({"acc": {"val_acc": val_acc}}, step=epoch_step)

        ckpt_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ckpt_state, last_ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_state, best_val_loss_ckpt_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch_step
            best_acc_model_state = copy.deepcopy(model.state_dict())

        if epoch_step in [10, 50]:
            cm = wandb.plot.confusion_matrix(
                y_true=y_true_all,
                preds=pred_all,
                class_names=["<=50K", ">50K"],
            )
            wandb.log({f"conf_mat_{epoch_step}": cm}, step=epoch_step)

            cm_counts = torch.zeros((2, 2), dtype=torch.float32)
            for y, p in zip(y_true_all, pred_all):
                cm_counts[y, p] += 1
            cm_norm = cm_counts / cm_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
            print(f"Row-normalized CM at epoch {epoch_step}:")
            print(cm_norm.tolist())

    if best_acc_model_state is not None:
        model.load_state_dict(best_acc_model_state)
        print(f"Loaded best-accuracy model from epoch {best_acc_epoch} (val_acc={best_val_acc:.4f})")

    model.eval()
    true_pos = 0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds = torch.argmax(outputs, dim=-1)
        true_pos += int((preds == torch.argmax(labels, dim=-1)).cpu().sum().item())
    acc = true_pos / len(val_dataset)
    print(f"Accuracy at the end of training: {acc:.4f}")
    wandb.log({"final": {"val_acc": acc}})

    num_classes = train_dataset[0][1].shape[0]
    target_n = 10
    counts = {c: 0 for c in range(num_classes)}
    table = wandb.Table(
        columns=["true_class", "sample_idx", "pred_class", "pred_prob", "correct", "input"]
    )

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            true_ids = torch.argmax(labels, dim=-1)

            for i in range(inputs.size(0)):
                y = int(true_ids[i].item())
                if counts[y] >= target_n:
                    continue

                p = int(pred_ids[i].item())
                conf = float(probs[i, p].item())
                correct = p == y

                counts[y] += 1
                x = inputs[i].detach().cpu().tolist()
                print(
                    f"class={y} sample={counts[y]}/{target_n} pred={p} prob={conf:.4f} "
                    f"correct={bool(correct)} input={x[:10]}"
                )
                table.add_data(y, counts[y], p, conf, bool(correct), x)

            if all(v >= target_n for v in counts.values()):
                break
    wandb.log({"val_examples_best_acc": table})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-share", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--resampled-dataset", action="store_true")
    if "CREATION_TIMESTAMP" in os.environ:
        timestamp = os.environ["CREATION_TIMESTAMP"]
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    parser.add_argument("--run-name", type=str, default=timestamp)
    args = parser.parse_args()
    main(args)
