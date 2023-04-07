from typing import NoReturn, Tuple, Dict
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data import PretextTaskDatasetWrapper
from src.trainer import BaseTrainer
from .config import AvgMeter

class ReconstructTrainer(BaseTrainer):

    """Trainer with pretext task"""

    def __init__(self, config: Dict):


        self._reconstruct = config['pretext']['reconstruct']
        self._jigsaw = config['pretext']['jigsaw']
        self._rotation = config['pretext']['rotation']
        super(ReconstructTrainer, self).__init__(config)

        self._absolute_loss = nn.MSELoss()
        self._loss_lambda = config['pretext']['lambda']

        self._name_postfix = 'reconstruct' 



    def train(self) -> NoReturn:
        dataset_wrapper = PretextTaskDatasetWrapper(batch_size=self._config['batch_size'],
                                                    reconstruct=self._reconstruct,
                                                    jigsaw=self._jigsaw,
                                                    rotation=self._rotation,
                                                    valid_size=self._config['dataset']['valid_size'],
                                                    input_size=eval(self._config['dataset']['input_shape']),
                                                    dataset=self._config['dataset']['dataset'])
        train_loader, valid_loader = dataset_wrapper.get_data_loaders()
        # create and if needed load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        # model = self._load_weights(model)
        model.to(self._device)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=3e-4, weight_decay=eval(self._config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader),
                                                               eta_min=0, last_epoch=-1)

        # create checkpoint and save
        checkpoint_folder = "runs/checkpoints"

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss_con = np.inf
        best_valid_loss_pre = np.inf


        for epoch_counter in range(1, self._config['epochs'] + 1):
            pretext_loss = AvgMeter()
            contrastive_loss = AvgMeter()
            # run training
            with tqdm(train_loader, unit="batch") as tepoch:
                for i, inputs in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch_counter} train")
                    optimizer.zero_grad()

                    # loss_contrastive, loss_pretext = self._step(model, inputs)
                    loss_pretext = self._step(model, inputs)

                    loss = loss_pretext 
                    # loss = loss_pretext + self._loss_lambda * loss_contrastive 
                    n_iter += 1

                    pretext_loss.update( loss_pretext.item())
                    # contrastive_loss.update(loss_contrastive.item())

                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss_pretext = pretext_loss.avg)

            # validation
            if epoch_counter % self._config['validate_every_n_epochs'] == 0:
                loss_pretext_valid = self._validate(model, valid_loader)
                # loss_contrastive_valid, loss_pretext_valid = self._validate(model, valid_loader)
                print(f"loss_pretext_valid: {loss_pretext_valid.item()}")
                # print(f"Contrastive: {loss_contrastive_valid.item()}, loss_pretext_valid: {loss_pretext_valid.item()}")
      
                valid_n_iter += 1

                
                # # save the best model
                # if best_valid_loss_con > loss_contrastive_valid:
                #     best_valid_loss_con = loss_contrastive_valid
                #     torch.save(model.state_dict(), checkpoint_folder + f'/model_best_contrastive_{epoch_counter}.pth')

                # save the best model
                if best_valid_loss_pre > loss_pretext_valid:
                    best_valid_loss_pre = loss_pretext_valid
                    torch.save(model.state_dict(), checkpoint_folder + f'/model_best_pretext_{epoch_counter}.pth')



            # schedule lr
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            torch.save(model.state_dict(), checkpoint_folder + f'/model_pretext.pth')

        # save final model
        torch.save(model.state_dict(), checkpoint_folder + '/model_final.pth')

    def _validate(self, model: nn.Module,
                  valid_loader: DataLoader) -> Tuple[float, float, float]:

        with torch.no_grad():
            # freeze weights
            model.eval()


            # loss_contrastive_total = 0.
            loss_pretext_total = 0.
            counter = 0

            for inputs in valid_loader:
                loss_pretext  = self._step(model, inputs)
                # loss_contrastive, loss_pretext,  = self._step(model, inputs)

                # loss_contrastive_total += loss_contrastive
                loss_pretext_total += loss_pretext
                counter += 1

            # loss_contrastive_total /= counter
            loss_pretext_total /= counter

        # unfreeze weights
        model.train()
        return loss_pretext_total

        # return loss_contrastive_total, loss_pretext_total


    def _step(self, model: nn.Module,
              inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:

        img1 = inputs[0].to(self._device)
        img2 = inputs[1].to(self._device)
        
        img_pretext = inputs[2].float().to(self._device)
        gt_pretext = inputs[3].float().to(self._device)
        
        # _, zis, _ = model(img1)
        # _, zjs, _ = model(img2)

        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)
        # loss_contrastive = self._nt_xent_criterion(zis, zjs)


        h, z, mask = model(img_pretext)
        loss_pretext = self._absolute_loss(mask, gt_pretext)
        
        return loss_pretext
        # return loss_contrastive, loss_absolute

    def _load_weights(self, model: nn.Module) -> nn.Module:
        checkpoint_file = Path(self._config['fine_tune_from'])

        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file)
            model.load_state_dict(state_dict)
            print(f'Loaded: {checkpoint_file}')
        else:
            print('Pre-trained weights not found. Training from scratch.')
        return model
