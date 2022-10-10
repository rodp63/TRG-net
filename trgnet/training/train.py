import torch

import trgnet.training.reference.utils as utils
from trgnet.training.reference.engine import evaluate, train_one_epoch
from trgnet.training.utils import load_training, save_training


def train(model, train_loader, validation_loader, epochs=1, load_epoch=-1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if load_epoch >= 0:
        model, optimizer = load_training(model, optimizer, load_epoch)

    model.to(device)
    for epoch in range(load_epoch + 1, load_epoch + 1 + epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, validation_loader, device)
        save_training(model, optimizer, epoch)
