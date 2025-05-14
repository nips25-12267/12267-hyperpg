import torch
import torch.nn as nn

from .backbones.backbone_factory import BackboneABC
from .model_factory import register_model
from .prototype_modules import pm_factory as pmf


class PPNConvNeck(torch.nn.Module):
    def __init__(self, backbone: BackboneABC, latent_channels: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=backbone.out_channels,
                out_channels=latent_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=latent_channels, out_channels=latent_channels, kernel_size=1
            ),
            # nn.LayerNorm((latent_channels, backbone.out_width, backbone.out_height)),
            nn.Sigmoid(),
            # nn.BatchNorm2d(latent_channels),
        )
        # for layer in self.convs:
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.kaiming_uniform_(layer.weight)

    def forward(self, feats: torch.Tensor):
        return self.convs(feats)


@register_model(name="protopnet")
class ProtoPNet(torch.nn.Module):
    def __init__(
        self,
        backbone: BackboneABC,
        num_classes: int,
        num_protos: int,
        latent_channels: int,
        pm_name: str,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone

        self.neck = PPNConvNeck(backbone, latent_channels)

        self.prototype_module = pmf.create_prototype(
            pm_name, num_classes, num_protos, latent_channels
        )

        self.clf = nn.Sequential(
            nn.MaxPool2d((backbone.out_width, backbone.out_height)),
            nn.Flatten(),
            nn.Linear(num_classes * num_protos, num_classes, bias=bias),
        )

    def forward(self, batch):
        feats = self.backbone(batch)
        feats = self.neck(feats)
        proto_d = self.prototype_module(feats)
        if type(proto_d) is tuple:
            # to handle if the prototype module returns both density and distance
            proto_density, proto_dist = proto_d
        else:
            proto_density = proto_d

        b, c, p, h, w = proto_density.shape

        pred = self.clf(proto_density.view(b, c * p, h, w))

        return pred, proto_d
