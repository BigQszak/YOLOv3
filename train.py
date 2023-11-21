import torch
import torch.optim as optim
import os

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

        with torch.cuda.amp.autocast_mode():
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


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES, config=model_config.config).to(
        config.DEVICE
    )
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=os.path.join(
            os.path.dirname(__file__), config.DATASET, "train.csv"
        ),
        test_csv_path=os.path.join(
            os.path.dirname(__file__), config.DATASET, "test.csv"
        ),
    )

    # Loading pretrained model
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    best_map = 0.0

    ### Proper trainign loop
    for epoch in range(config.NUM_EPOCHS):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 30)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
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


if __name__ == "__main__":
    main()
