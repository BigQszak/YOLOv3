import torch
import torch.nn as nn


def test(model):
    num_classes = 20
    IMAGE_SIZE = 416
    model = model(num_classes=num_classes)
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
