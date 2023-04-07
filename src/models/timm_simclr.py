
from typing import Tuple
import timm
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.nn import functional as F

class TimmModelBase(nn.Module):
    def __init__(self, model_name='timm-efficientnet-b3', out_dim=128, in_channels=3, classes=1):
        super().__init__()

        self.backbone = timm.create_model(model_name='efficientnet_b3', pretrained=True, features_only=True)
        self.gap = nn.AdaptiveAvgPool2d(1) 
        self._l1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.ReLU(),
            nn.Dropout()
        )

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)[-1]
        h = self.gap(h)
        # h = h.squeeze()

        z = self._l1(h)

        return h, z