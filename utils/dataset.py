import numpy as np
import os 
import pandas as pd 
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
# from utils import (
#     iou_width_height as iou,
#     non_max_supression as nms
# ) TO BE IMPLEMENTED

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, csv_file: any, img_dir: str, label_dir: str, anchors: list, image_size=416, S=[13, 26, 52], C=20, transform=None) -> None:
        super().__init__()
        """ 
        S => grid sizes for different scale predictions
        C => number of classes
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # three different scales, three anchors each
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C 
        self.ignore_iou_thresh = 0.5

    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index) -> Any:
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist() # [class, x, y, w, h]
     
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes) # albumentations
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        # [probability of object, x, y, w, h, class] => 6


