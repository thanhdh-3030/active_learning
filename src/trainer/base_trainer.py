from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn, Dict
import shutil

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.losses import NTXentLoss
from src.models import ResNetSimCLR, ScatSimCLR, TimmModelBase
from src.evaluation import LogisticRegressionEvaluator
from src.data import EmbeddingExtractor
from src.data.datasets import NUM_CLASSES


def get_device(device: int) -> str:
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    return device


class BaseTrainer(ABC):

    """Abstract base trainer class"""

    EMBEDDINGS_MODELS = ['resnet18', 'resnet50', 'timm-efficientnet-b3', 'timm-efficientnet-b4','timm-efficientnet-b5',
                         'scatsimclr8', 'scatsimclr12', 'scatsimclr16', 'scatsimclr30', 'scatsimclr45']

    def __init__(self, config: Dict):
        self._config = config
        self._device = get_device(self._config['device'])

        self._nt_xent_criterion = NTXentLoss(self._device, config['batch_size'], **config['loss'])

    @abstractmethod
    def train(self):
        pass

    def evaluate(self) -> float:
        # load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        model = self._load_weights(model)
        model.to(self._device)
        model.eval()

        score = self._test_classification(model)
        return score

    @abstractmethod
    def _step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _load_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def _validate(self, *args, **kwargs):
        pass

    def _test_classification(self, model) -> float:
        """Tests classification with extracted features

        Args:
            model: CLR model to use to compute embeddings for classification

        Returns:
            float: classification accuracy
        """

        model.eval()

        input_size = eval(self._config['dataset']['input_shape'])
        dataset = self._config['dataset']['dataset']
        epochs = 100

        extractor = EmbeddingExtractor(model, device=self._device,
                                       dataset=dataset,
                                       input_size=input_size, batch_size=self._config['batch_size'])
        train_data, train_labels, test_data, test_labels = extractor.get_features()

        evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1],
                                                n_classes=NUM_CLASSES[dataset],
                                                device=self._device, batch_size=64)
        accuracy = evaluator.run_evaluation(train_data, train_labels, test_data, test_labels, epochs)
        return accuracy

    def _get_embeddings_model(self, model_name: str) -> nn.Module:

        if model_name not in self.EMBEDDINGS_MODELS:
            raise ValueError('Unsupported model')

        out_dim = self._config['model']['out_dim']

        if 'resnet' in model_name:
            return ResNetSimCLR(base_model=model_name, out_dim=out_dim)

        if 'scatsimclr' in model_name:
            blocks = int(model_name[10:])
            input_size = eval(self._config['dataset']['input_shape'])

            J = self._config['model']['J']
            L = self._config['model']['L']

            return ScatSimCLR(J=J, L=L, input_size=input_size,
                              res_blocks=blocks, out_dim=out_dim)
        
        if 'efficientnet' in model_name:
            return TimmModelBase(model_name=model_name,out_dim=out_dim)
