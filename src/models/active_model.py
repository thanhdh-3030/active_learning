import os
import torch
import glob
import numpy as np
# import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset

# Apply WanDB
import wandb
import pytorch_lightning as pl

"""## Define segmentation model """

class ActiveSegmentationModel(pl.LightningModule):

    def __init__(self, arch='', encoder_name='', in_channels=3, out_classes=1, checkpoint_path='', labeled_dataloader=None,save_model=True, 
                 unlabeled_dataloader=None,lr=1e-4,memory_size=256,temperature=0.2,proj_dim=256,**kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        
        self.save_model = save_model

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # max iou of origin model and momentum model  
        self.m_max_iou = 0
        self.checkpoint_path = checkpoint_path
        self.beta = 0.3
        
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        if self.unlabeled_dataloader == None:
            self.is_semi = False
        else: self.is_semi = True
        self.lr=lr
        self.queue_len=memory_size
        self.proj_dim=proj_dim
        self.temperature=temperature
        # define queue for memory bank
        self.register_buffer("queue",torch.randn(out_classes,self.queue_len,self.proj_dim)) # (C,M,D)
        self.queue=torch.nn.functional.normalize(self.queue,p=2,dim=2)
        self.register_buffer('queue_ptr',torch.zeros(out_classes,dtype=torch.long)) # (C,)
    
    # dequeue and enqueue for memory bank
    def _dequeue_and_enqueue(self,keys,labels,
                             category,bs):
        if category not in labels:
            return
        keys=keys[list(labels).index(category)]
        ptr=int(self.queue_ptr[category])
        self.queue[category,:,ptr]=keys
        self.queue_ptr[category]=(ptr+bs)%self.queue_len

    # compute region embedding
    def construct_region(self,feature,predict):
        """ Construct region embeddings for all category appear in predicted mask.
        Args:
            feature (torch.Tensor): The embedding after upsampling with shape
                (B,D,H,W) where D is embedding dim.
            predict (torch.Tensor): Predicted mask with shape
                (B,C,H,W).
        Returns:
            torch.Tensor: (C,D) where C is the number of category in this base.
            torch.Tensor: (C,)
        """
        bs=feature.shape[0]
        embedding_dim=feature.shape[1]
        predict=predict.squeeze().view(bs,-1)
        categories=torch.unique(predict)
        feature=feature.view(bs,embedding_dim,-1)
        region_embeddings=torch.zeros([categories,embedding_dim])
        for category in categories:
            cat_feature= feature[:,predict==category].mean(1).unsqueeze(0)
            region_embeddings[category]=cat_feature
        return region_embeddings,categories.cuda()
    
    def _compute_positive_contrastive_loss(self,keys,appeared_categories):
        """ Calculate contrastive loss enfoces the embeddings of same class
            to be close and different class far away.
        """
        contrast_loss=0
        for cls_ind in appeared_categories:
            query=keys[list(appeared_categories).index(cls_ind)] # (1,D)
            positive_keys= self.queue[cls_ind].clone().detach() # (M,D)
            all_ids=[i for i in range (2)] # all classes
            neg_ids=all_ids.copy().remove(cls_ind)
            negative_keys=self.queue[neg_ids] # 
        return 
    def forward(self, image):
        # normalize image here
        mask = self.model(image)
        return mask
    def forward_encoder(self,image):
        embedding=self.model.encoder(image)
        return embedding
    def forward_decoder(self,image):
        features=self.model.encoder(image)
        embedding=self.model.decoder(*features)
        return embedding
    def forward_dropout(self, image, n_drop=10):
        # normalize image here
        # self.model.train()
        b, c, w, h = image.shape
        
        probs = torch.zeros([b, 1, w, h]).cuda(image.get_device())
        
        for _ in range(n_drop):
            mask = self.model(image)
            probs += mask.sigmoid()
        probs = probs / n_drop
        return probs
    def forward_dropout_split(self,image,n_drop=10):
        self.model.train()
        b, _, w, h = image.shape
        probs=torch.zeros([n_drop,b,1,w,h])
        for i in range(n_drop):
            pred=self.model(image)
            prob=pred.sigmoid()
            probs[i]=prob.cpu()
        return probs        
    def compute_loss(self, image, mask):
        assert image.ndim == 4
        assert mask.ndim == 4
        
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        result = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "loss": loss
        }
        return result

    def shared_step(self, batch, stage):
        # If not have unlabeled data, the batch is only have labeled data
        batch_labeled = batch
        
        # if stage == 'train':
        if self.is_semi:
            batch_labeled = batch['labeled']
        
        # Batch of labeled
        l_image = batch_labeled["image"]
        l_mask = batch_labeled["mask"]
        
        l_result = self.compute_loss(l_image, l_mask)
        l_loss = l_result['loss']
        # Predefine for unlabeled loss 
        u_result = l_result
        u_result['loss'] = 0
        u_loss = u_result['loss']
        
        # Batch of unlabeled 
        if self.is_semi and stage == 'train':
            batch_unlabeled = batch['unlabeled']
            u_image = batch_unlabeled["image"]
            u_mask = batch_unlabeled["mask"]
            u_result = self.compute_loss(u_image, u_mask)
            u_loss = u_result['loss']
        
        # Compute total loss 
        loss = l_loss + self.beta * u_loss

        
        # Append loss to result
        result = l_result
        result['loss'] = loss
        loss = result['loss']

        return result
    
    def calculate_predictions(self, batch):
        image = batch["image"]
        mask = batch["mask"]
        logits_mask = self.forward(image)
        return logits_mask
        

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])    
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        
        if stage == 'valid':
            # Save best checkpoint
            if per_image_iou > self.m_max_iou:
#                 print('\nSave origin model with checkpoint loss = {}'.format(per_image_iou))
#                 torch.save(self.model, self.checkpoint_path + '.ckpt')
                self.m_max_iou = per_image_iou

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
        }
        
        if self.save_model:
            torch.save(self.model, self.checkpoint_path)
        

    def train_dataloader(self):
        if self.is_semi:
            loaders = {"labeled": self.labeled_dataloader, "unlabeled": self.unlabeled_dataloader}
        else:
            loaders = {"labeled": self.labeled_dataloader}

        return loaders
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
        return {
        "optimizer": optimizer,
        "lr_scheduler":scheduler}