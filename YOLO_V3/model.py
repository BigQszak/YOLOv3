import torch
import torch.nn as nn
import lightning as pl


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, batch_norm=True, **kwargs
    ) -> None:  # **kwargs -> kernel_size, stride, padding etc
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1)
        self.use_batch_norm = (
            batch_norm  # we do not want to use bn & leaky in our scale predictions
        )

    def forward(self, x):
        if self.use_batch_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        for repeat in range(num_repeats):
            self.layers.append(
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
            )

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = (
                    layer(x) + x
                )  # num_channels has not changed from before the block so we can simply add feature maps together
            else:
                x = layer(x)
        return x


class Prediction(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
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

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
        # N(batch) x 13x13(grid at one scale) x 3(anchors per cell) x  5(bbox)+num_classes


class YOLOv3(nn.Module):
    pass


if __name__ == "__main__":
    pass
