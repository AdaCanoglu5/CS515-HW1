import copy
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


LOG_FIELDS = [
    "run_name",
    "seed",
    "hidden_sizes",
    "activation",
    "dropout",
    "use_bn",
    "bn_position",
    "lr",
    "scheduler",
    "step_size",
    "gamma",
    "tmax",
    "weight_decay",
    "l1_lambda",
    "batch_size",
    "epochs",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "best_val_acc_so_far",
    "best_epoch_so_far",
    "lr_current",
    "test_acc",
    "test_per_class",
]


def append_csv_row(params, row):
    log_csv = params.get("log_csv")
    if not log_csv:
        return

    csv_dir = os.path.dirname(log_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    write_header = not os.path.exists(log_csv) or os.path.getsize(log_csv) == 0
    full_row = {k: "" for k in LOG_FIELDS}
    full_row.update({
        "run_name": params.get("run_name", ""),
        "seed": params["seed"],
        "hidden_sizes": ",".join(map(str, params["hidden_sizes"])),
        "activation": params["activation"],
        "dropout": params["dropout"],
        "use_bn": int(params["use_bn"]),
        "bn_position": params["bn_position"],
        "lr": params["learning_rate"],
        "scheduler": params["scheduler"],
        "step_size": params["step_size"],
        "gamma": params["gamma"],
        "tmax": params["tmax"],
        "weight_decay": params["weight_decay"],
        "l1_lambda": params["l1_lambda"],
        "batch_size": params["batch_size"],
        "epochs": params["epochs"],
    })
    full_row.update(row)

    with open(log_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)


def get_loaders(params):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval, l1_lambda):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def run_training(model, params, device):
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
    scheduler = None
    if params["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=params["step_size"], gamma=params["gamma"]
        )
    elif params["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params["tmax"]
        )

    best_acc     = 0.0
    best_epoch   = 0
    best_weights = None
    save_dir = os.path.dirname(params["save_path"])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, params["log_interval"],
                                          params["l1_lambda"])
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        lr_current = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_epoch   = epoch
            best_weights = copy.deepcopy(model.state_dict())  # snapshot in memory
            torch.save(best_weights, params["save_path"])      # persist to disk
            print(f" Saved best model (val_acc={best_acc:.4f})")

        append_csv_row(params, {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc_so_far": best_acc,
            "best_epoch_so_far": best_epoch,
            "lr_current": lr_current,
        })

    # Restore best weights into the model before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
