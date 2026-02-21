import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


# ==================================================
# CNN BASELINE — EXACT MATCH TO TRAINED WEIGHTS
# ==================================================
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # features.3
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # NOTE: no adaptive pooling, full flatten
        self.classifier = nn.Sequential(
            nn.Linear(200704, 256),   # classifier.0
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # classifier.3
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==================================================
# VISION TRANSFORMER MODEL — SAME AS PHASE 4
# ==================================================
class ViTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT
        self.vit = vit_b_16(weights=weights)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
