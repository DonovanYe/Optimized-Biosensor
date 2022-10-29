import torch
import torch.nn as nn

class ClassificationAccuracy(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return "cls_acc"

    @property
    def mode(self):
        return "max"

    @property
    def task(self):
        return "cls"

    def forward(self, pred, gt):
        return torch.mean((pred.argmax(dim=1) == gt).type(torch.float32))