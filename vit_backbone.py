import torch
import torch.nn as nn
from model import ViTModel


class ViTFeatureExtractor(nn.Module):
    def __init__(self, weight_path):
        super().__init__()

        self.vit = ViTModel(num_classes=4)
        self.vit.load_state_dict(torch.load(weight_path, map_location="cpu"))

        # remove classifier head
        self.vit.vit.heads.head = nn.Identity()

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.vit(x)
