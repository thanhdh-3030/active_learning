import numpy as np
from src.trainer.config import *

epsilon = 1e-7
def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + epsilon)


def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    # print(f"scores: dice={mean_dice}, miou={mean_iou}, precision={mean_precision}, recall={mean_recall}\n")

    return (mean_iou, mean_dice, mean_precision, mean_recall)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        



import torch

def active_contour_loss(y_true, y_pred, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    # length term
    delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
    delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
    
    delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
    delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
    delta_pred = torch.abs(delta_r + delta_c) 

    epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
    
    C_in  = torch.ones_like(y_pred)
    C_out = torch.zeros_like(y_pred)

    region_in  = torch.mean( y_pred     * (y_true - C_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean( (1-y_pred) * (y_true - C_out)**2 ) 
    region = region_in + region_out
    
    loss =  weight*lenth + region

    return loss


def get_testdataset(dataset):
    return ActiveDataset(
        image_paths=glob.glob('{}/{}/images/*'.format('TestDataset', dataset)), 
        gt_paths=glob.glob('{}/{}/masks/*'.format('TestDataset/', dataset)), 
        trainsize=trainsize, 
        transform=val_transform
    )

from src.evaluation.metric import get_scores
from tabulate import tabulate

def full_val(model, budget_size, use_wandb=True, device='cuda'):
    print("#" * 20)
    model.eval()
    model.to(device)
    
    dataset_names = os.listdir('TestDataset/')
    table = []
    headers = ['Dataset', 'IoU', 'Dice']
    ious, dices = AvgMeter(), AvgMeter()

    for dataset_name in dataset_names:
        tmp_dataset = get_testdataset(dataset_name)
        test_loader = DataLoader(tmp_dataset, batch_size=1, shuffle=False, num_workers=4)   

        gts = []
        prs = []
        for i, pack in enumerate(test_loader, start=1):
            image, gt = pack["image"], pack["mask"]
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            image = image.to(device)

            res = model(image)
            # res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()
            gts.append(gt)
            prs.append(pr)
        mean_iou, mean_dice, _, _ = get_scores(gts, prs)
        ious.update(mean_iou)
        dices.update(mean_dice)
        if use_wandb:
            wandb.log({f'{dataset_name}_dice': mean_dice, 'num_sample' : budget_size})
        table.append([dataset_name, mean_iou, mean_dice])
    table.append(['Total', ious.avg, dices.avg])
    if use_wandb:
        wandb.log({f'total_dice': dices.avg, 'num_sample' : budget_size})
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return ious.avg, dices.avg