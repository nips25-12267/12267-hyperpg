import torch
import torch.nn as nn

from .backbones.backbone_factory import BackboneABC
from .model_factory import register_model


@register_model(name="baseline")
class BaselineModel(torch.nn.Module):
    def __init__(
        self,
        backbone: BackboneABC,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.backbone = backbone

        self.neck = nn.Identity()
        self.prototype_module = nn.Identity()

        clf_channels = backbone.out_channels

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(clf_channels, num_classes)
        )

    def forward(self, batch):
        pred = self.backbone(batch)
        pred = self.neck(pred)
        pred = self.prototype_module(pred)

        pred = self.clf(pred)
        return pred, None
