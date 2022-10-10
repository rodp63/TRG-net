import os

import torch


def load_training(model, optimizer=None, epoch=7, name="trgnet"):
    user_path = os.path.expanduser("~")
    pt_path = os.path.join(user_path, f".trgnet/checkpoints/{name}-{epoch}.pt")
    checkpoint = torch.load(pt_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loading model at epoch {epoch}")
    if optimizer:
        return model, optimizer
    return model


def save_training(model, optimizer, epoch, name="trgnet"):
    user_path = os.path.expanduser("~")
    pt_path = os.path.join(user_path, f".trgnet/checkpoints/{name}-{epoch}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pt_path,
    )
