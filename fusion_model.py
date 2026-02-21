import torch
import torch.nn as nn
from cnn_backbone import CNNFeatureExtractor
from vit_backbone import ViTFeatureExtractor


class FusionModel(nn.Module):
    def __init__(self, cnn_weights, vit_weights, num_classes=4):
        super().__init__()

        self.cnn = CNNFeatureExtractor(cnn_weights)
        self.vit = ViTFeatureExtractor(vit_weights)

        fusion_dim = 512 + 768  # CNN + ViT

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(fused)
