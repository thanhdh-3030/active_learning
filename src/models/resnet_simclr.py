from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from segmentation_models_pytorch.encoders import get_encoder


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model: str,
                 out_dim: int,
                 pretrained: bool = True):
        """
        Args:
            base_model: base model to be used as feature extractor

            out_dim: size of output vector Z

            pretrained: if True, then pretrained weights will be downloaded
        """

        super(ResNetSimCLR, self).__init__()

        self.model_dict = {'resnet18': models.resnet18(pretrained=pretrained),
                             'resnet50': models.resnet50(pretrained=pretrained),
                             'efficientnet-b4':get_encoder(name='timm-efficientnet-b4', weights='imagenet')}

        if base_model not in self.model_dict.keys():
            raise ValueError(f'Invalid model name. It can be one '
                             f'of: [{", ".join(self.model_dict.keys())}]')

        model = self.model_dict[base_model]
        if 'efficientnet' in base_model:
            num_ftrs = 448
            self._features = nn.Sequential(*list(list(model.children())[:-4] + list(model.children())[-1:]))
        else:
            num_ftrs = model.fc.in_features
            self._features = nn.Sequential(*list(model.children())[:-1])

        # projection MLP
        self._l1 = nn.Linear(num_ftrs, num_ftrs)
        self._l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._features(x)
        h = h.squeeze()

        z = self._l1(h)
        z = F.relu(z)
        z = self._l2(z)
        return h, z
