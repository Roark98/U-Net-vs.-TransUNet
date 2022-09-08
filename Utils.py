from torch.utils.data import Dataset
import torch
import os
import cv2
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def dice_score(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union

class CustomDataset(Dataset):
  def __init__(self, pathImg, pathLbl, images, labels, transform=None):

    self.images = images
    self.labels = labels
    self.pathImg = pathImg
    self.pathLbl = pathLbl

    self.transform = transform
    
  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_file_path = os.path.join(self.pathImg, self.images[idx])
    lbl_file_path = os.path.join(self.pathLbl, self.labels[idx])

    img = cv2.imread(img_file_path, 0)
    mask = cv2.imread(lbl_file_path, 0)/255

    #Pytorch Transformations
    img = self.transform(img)
    mask = self.transform(mask)

    return img, mask

def get_model(model_name):
  if model_name=='UNet':
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=1, out_channels=1, init_features=32, pretrained=False)
  elif model_name=='TransUNet':
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3

    return ViT_seg(config_vit, img_size=320, num_classes=config_vit.n_classes)
