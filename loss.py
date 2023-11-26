import torch
import torch.nn as nn

from metrics import intersection_over_union as iou


class YOLOLoss(nn.Module):
    """
    Calculating loss for a single scale prediction
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants
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
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_predictions = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5] * anchors),
            ],
            dim=-1,
        )
        ious = iou(box_predictions[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(
            self.sigmoid(predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])
        )

        # Box coordinate loss
        predictions[..., 1:3] = self.sigmoid(
            predictions[..., 1:3]
        )  # x & y between 0 and 1
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long())
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_no_obj * no_obj_loss
            + self.lambda_class * class_loss
        )
