import torch
from collections import Counter
from tqdm import tqdm

import utils


def iou_width_height(boxes1, boxes2):
    """
    Args:
        boxes1 (tensor): width and height of the first bounding box
        boxes2 (tensor): width and height of the second bounding box

    Returns:
        tensor: Intersection over union of the corresponding boxes
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )  # minimum of width & hight
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )  # summing the areas of both boxes and subtracting the intersection
    return intersection / union


def intersection_over_union(
    boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str = "midpoint"
):
    """
    This function calculates intersection over union (iou) given prediction boxes
    and target boxes.
    IoU is a metric for evaluation bounding box predictions - that is how closed to
    the ground truth a specific predicted box is.
        Intersection - measures the overlap between boxes - common area.
        Union - measures the combined area of two boxes
    It is a value between 0 and 1. The higher the value the better the prediction.
    IoU > 0.5 is a good threshold.
    IoU > 0.7 is a very good threshold.
    IoU > 0.9 is an excellent threshold.

    Remember that in computer graphics the origin is in the top left corner (growing to the right and downwards).

    Depending of the bounding box format we get the bounding box values differently.

    Intersection area:
    x1 is the top let corner of the "intersection box", thus must be the largest value of the two boxes corners
    x2 is the bottom right corner of the "intersection box", thus must be the smallest value of the two boxes corners
    Similar with y values, y1 is top left and y2 is bottom right.

    Args:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        # Calculating the corners of both boxes based on their centerpoint, width and hight
        # Division by 2 of width and hight is to get the corners (midpoint - half of the width and hight)
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        # [..., ] means all the previous dimensions (bathch_size - all the predictions)
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        # we want to maintain the tensor dimensionality (N, 1)

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # Intersection Area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # .clamp(0) covers the case when the boxes do not inteersect at all

    # Union Area
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (
        box1_area + box2_area - intersection + 1e-6
    )  # 1e-6 for numerical stability


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="corners"):
    """
    Performs Non Max Suppression given list of bounding boxes.
    NMS is an algorithm that removes overlapping bounding boxes.
    It utilises the Intersection over Union (IoU) metric to find overlapping.

    Detection results in plenty of bounding boxes, but we want to end up in one - best one
    per object.

    We can start by discarding boxes below some probabiliy threshold (of having an object inside)
    For each object class in the image:
        Each box assosiated with it comes with probabilty of the object being in that box.
        We take the one with the highest one and calculate its iou with the second best one.
        If the iou is higher than the threshold it means that those two boxes are responsible for
        detecting the same object, however one is better than the other.
        We discard the "weaker" box.
        If the iou is lower than the threshold it means the sexond box is responsible for detecting
        other instance of that class and we skip it.
        We repeat the process until we run out of boxes.
        At the end we "save" out main box and continue with the process with the new box
        with highest probabilty.


    Args:
        bboxes (list): list of lists, each list represents one bounding box
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS for a specific IoU threshold
    """

    assert type(bboxes) == list, "Argument bboxes should be a list"

    # removing boxes with prob less than threshold
    bboxes = [box for box in bboxes if box[1] > prob_threshold]

    # sort boxes by probability descending
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    # until we run out of boxes
    while bboxes:
        # choose the box with the highest prob score
        chosen_box = bboxes.pop(0)

        # compare out main box with
        bboxes = [
            box
            for box in bboxes
            if box[0]
            != chosen_box[
                0
            ]  # if the boxes are of different classes we do not want to compare them - keep the box
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,  # calculating iou between boxes, taking only box dimensions
            )
            < iou_threshold  # if the iou is lower than the threshold we keep the box
        ]

        bboxes_after_nms.append(
            chosen_box
        )  # we append out list of cleaned boxes with the chosen one

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Function for calculates mean average precision (mAP) metric.

    Args:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    """
    Converts the predicted boxes from the test data to a readable format
    """
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = utils.cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = utils.cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes
