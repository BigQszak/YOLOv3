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

        for box in bboxes: # which anchor is responsible for which scale predictions - highest iou
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # we are sending width & height of the bbox
            """
             TO_DO: implement iou()
            """
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # best anchors
            x, y, width, height, class_label = box # from loaded box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale 
                anchors_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                grid_y, grid_x = int(S*y), int(S*x)
                anchor_chosen = targets[scale_idx][anchors_on_scale, grid_y, grid_x, 0]

                if not anchor_chosen and not has_anchor[scale_idx]:
                    targets[scale_idx][anchors_on_scale, grid_y, grid_x, 0] = 1
                    x_cell, y_cell = S*x - grid_x, S*y - grid_y # specific location within a grid cell
                    width_cell, height_cell = (
                        width*S, 
                        height*S
                        )
                    
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchors_on_scale, grid_y, grid_x, 5] = box_coordinates 
                    targets[scale_idx][anchors_on_scale, grid_y, grid_x, 5] = int(class_label)

                    """ 
                    Change has_anchor to True
                    """
                elif not anchor_chosen and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchors_on_scale, grid_y, grid_x, 0] = -1 # ignore prediction

        return image, tuple(targets)
 











