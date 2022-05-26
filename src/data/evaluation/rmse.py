import torch


def rmse(prediction, ground_truth):
    return torch.sqrt(torch.pow(prediction - ground_truth, 2).sum() / len(prediction))

