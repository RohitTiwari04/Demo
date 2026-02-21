import torch
import torch.nn as nn
from model import CNNBaseline


class CNNFeatureExtractor(nn.Module):
    def __init__(self, weight_path):
        super().__init__()

        self.cnn = CNNBaseline(num_classes=4)
        self.cnn.load_state_dict(torch.load(weight_path, map_location="cpu"))

        self.features = self.cnn.features

        # 🔥 projection to make fusion GPU-safe
        self.proj = nn.Linear(200704, 512)

        for p in self.cnn.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)
