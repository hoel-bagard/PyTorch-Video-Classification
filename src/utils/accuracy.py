import torch
import torch.nn as nn


def get_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, max_batches: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = 0
    for step, batch in enumerate(dataloader, start=1):
        imgs, labels = batch["video"].float(), batch["label"]
        predictions = model(imgs.to(device))
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        acc += torch.mean(torch.eq(labels.to(device), torch.argmax(predictions, dim=-1)).float())
        if step >= max_batches:
            break
    return acc / step
