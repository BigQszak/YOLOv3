import torch
import torch.nn as nn


from utils import intersection_over_union as iou

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #Constants  
        self.lambda_class = 1
        self.lambda_no_obj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions: torch.tensor, target: torch.tensor, anchors: list):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # No object loss
        no_obj_loss = self.bce(
            predictions[..., 0:1][no_obj], (target[..., 0:1][no_obj])
        )

        # Object loss
        

        # Box coordinate loss

        # Class loss



