import torch


def get_accuracy(labels: torch.Tensor, predictions: torch.Tensor):
    """
    Args:
        labels: labels, are expected to have shape [batch]
        predictions: expected to have shape [batch, sequence, nb_classes]
    """
    predictions = torch.argmax(predictions, dim=-1)
    # predictions = predictions[:, -1]    # If a prediction was made for every frame
    accuracy = torch.sum(torch.eq(labels, predictions)).detach().numpy() / labels.shape[0]

    return accuracy
