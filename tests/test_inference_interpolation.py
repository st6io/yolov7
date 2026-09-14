from unittest.mock import patch

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.common import autoShape
from utils.datasets import letterbox


class FakeModel(nn.Module):
    names = ["waste"]
    stride = torch.tensor([32])

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.input = None

    def forward(self, image, augment=False, profile=False):
        self.input = image.detach().clone()
        return (torch.zeros((image.shape[0], 0, 6), device=image.device),)


def test_letterbox_keeps_bilinear_as_default():
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    with patch("utils.datasets.cv2.resize", wraps=cv2.resize) as resize:
        letterbox(image, new_shape=(8, 8), auto=False)

    assert resize.call_args.kwargs["interpolation"] == cv2.INTER_LINEAR


def test_autoshape_uses_nearest_neighbor_for_inference():
    model = FakeModel()
    inference = autoShape(model)
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    with patch("models.common.letterbox", wraps=letterbox) as resize_and_pad:
        inference(image, size=8)

    assert resize_and_pad.call_args.kwargs["interpolation"] == cv2.INTER_NEAREST
