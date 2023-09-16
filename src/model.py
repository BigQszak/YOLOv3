import torch
import torch.nn as nn

import lightning as pl
from model_config import return_model_config

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, batch_norm=True, **kwargs
    ) -> None:  # **kwargs -> kernel_size, stride, padding etc
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.use_batch_norm = batch_norm
        # we do not want to use bn & leaky in our scale predictions

    def forward(self, x):
        if self.use_batch_norm:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvBlock(
                        in_channels=channels, out_channels=channels // 2, kernel_size=1
                    ),
                    ConvBlock(
                        in_channels=channels // 2,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
                # num_channels has not changed from before the block so we can simply add feature maps together
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.prediction = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=2 * in_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBlock(
                in_channels=2 * in_channels,
                out_channels=3 * (num_classes + 5),
                # for each cell = 3 anchors, for every box we have outputs for probability of each class, 5 = bbox parameters [probability,x,y,w,h]
                batch_norm=False,
                kernel_size=1,
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.prediction(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        # N(batch) x 13x13(grid at one scale) x 3(anchors per cell) x  5(bbox)+num_classes


class YOLOv3(nn.Module):
    def __init__(self, config: list, in_channels=3, num_classes=20) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.config = config
        self.layers = self._create_layers()

    def _create_layers(self):
        layers = nn.ModuleList()
        in_channels = (
            self.in_channels
        )  # local variable, used to create consecutive layers

        for module in self.config:
            if isinstance(module, tuple):  # conv layer
                out_channels, kernel_size, stride = module
                layers.append(
                    ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels  # for next layer

            elif isinstance(module, list):  # residual block
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(channels=in_channels, num_repeats=num_repeats)
                )

            elif isinstance(module, str):  # yolo prediction or upsampling
                if module == "S":
                    layers += [
                        ResidualBlock(
                            channels=in_channels, use_residual=False, num_repeats=1
                        ),
                        ConvBlock(
                            in_channels=in_channels,
                            out_channels=in_channels // 2,
                            kernel_size=1,
                        ),
                        ScalePrediction(
                            in_channels=in_channels // 2, num_classes=self.num_classes
                        ),
                    ]  # concatenate list with another list of modules (layers)
                    in_channels = (
                        in_channels // 2
                    )  # wa need to continue later from ConvBlock

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    in_channels = (
                        in_channels * 3
                    )  # we will be concatenating here, so we need to match channels dimensions

        return layers

    def forward(self, x):
        outputs = []  # different scales predictions
        route_connection = []  # skip connection origin

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(
                    layer(x)
                )  # catching the outputs of each scale predictions to combine later
                continue  # we will continue to move forward through the network from the branching point where we split into prediction branch

            x = layer(x)

            if (
                isinstance(layer, ResidualBlock) and layer.num_repeats == 8
            ):  # this is where we are creating skip connection root
                route_connection.append(
                    x
                )  # we are saving this level output to later combine it with deeper layers

            elif isinstance(layer, nn.Upsample):
                x = torch.cat(
                    [x, route_connection[-1]], dim=1
                )  # when we upsample this is where we want to concatenate with the last skip connection
                route_connection.pop()  # removing that last skip connection

        return outputs

def test(config: list):
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes, config=config)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))  # batch_size, RGB, spatial_dim
    out = model(x)

    assert model(x)[0].shape == (
        2,
        3,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        num_classes + 5,
    )
    assert model(x)[1].shape == (
        2,
        3,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        num_classes + 5,
    )
    assert model(x)[2].shape == (
        2,
        3,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
        num_classes + 5,
    )
    print("Test completed successfully")

if __name__ == "__main__":

    test(return_model_config())

