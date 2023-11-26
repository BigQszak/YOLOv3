import torch
import torch.optim as optim
import os
import argparse

import config
import model_config
from model import YOLOv3
from tqdm import tqdm
from metrics import mean_average_precision, get_evaluation_bboxes
from utils import (
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    plot_couple_examples,
    cells_to_bboxes,
)
from loss import YOLOLoss
from dataset import get_loaders

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():  # with torch.cuda.amp.autocast_mode():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)


def main(args):
    model = YOLOv3(
        num_classes=20 if args["dataset"] == "PASCAL_VOC" else 80,
        config=model_config.config,
    ).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"]
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=os.path.join(
            os.path.dirname(__file__), args["dataset"], "train.csv"
        ),
        test_csv_path=os.path.join(
            os.path.dirname(__file__), args["dataset"], "test.csv"
        ),
    )

    # Loading pretrained model
    if args["load_model"]:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, args["learning_rate"])

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    best_map = 0.0

    ### Proper trainign loop
    for epoch in range(args["num_epochs"]):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        print(f"Epoch {epoch + 1}/{args['num_epochs']}")
        print("-" * 30)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=args["conf_threshold"])

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=args["nms_iou_thresh"],
                anchors=config.ANCHORS,
                threshold=args["conf_threshold"],
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=args["map_iou_thresh"],
                box_format="midpoint",
                num_classes=20 if args["dataset"] == "PASCAL_VOC" else 80,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

        ### Saving model
        if config.SAVE_MODEL:
            save_checkpoint(
                model, optimizer, filename=f"epoch_{epoch}_checkpoint.pth.tar"
            )

            # save best checkpoint
            if mapval > best_map:
                best_map = mapval
                save_checkpoint(model, optimizer, filename=f"best_checkpoint.pth.tar")


def parse_arg():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=config.NUM_WORKERS,
        help="Multi-process data loading with the specified number of loader worker processes",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config.IMAGE_SIZE,
        help="Input size image for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET,
        help="Name of the dataset you want to use for training: PASCAL_VOC or COCO",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=config.CONF_THRESHOLD,
        help="Confidence threshold - probability of object being detected",
    )
    parser.add_argument(
        "--map_iou_thresh",
        type=float,
        default=config.MAP_IOU_THRESH,
        help="Threshold for mean average precision calculation",
    )
    parser.add_argument(
        "--nms_iou_thresh",
        type=float,
        default=config.NMS_IOU_THRESH,
        help="Threshold for nsm calculation",
    )
    parser.add_argument(
        "--load_model",
        action="store_true",
        help="Option to load pretrained model (default is False)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = parse_arg()
    main(vars(arguments))
