import os

import torch


def load_training(model, optimizer=None, epoch=7):
    file_path = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(file_path, f"checkpoints/model{epoch}.pt")
    checkpoint = torch.load(pt_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loading model at epoch {epoch}")
    if optimizer:
        return model, optimizer
    return model


def save_training(model, optimizer, epoch):
    file_path = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(file_path, f"checkpoints/model{epoch}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pt_path,
    )
