import torch
from torch import nn

from . import pm_factory as pmf


class _L2PrototypeModule(pmf.PrototypeModule):
    def __init__(self, num_classes: int, num_prototypes: int, latent_channels: int):
        super().__init__(num_classes, num_prototypes, latent_channels)
        nn.init.trunc_normal_(self.prototypes)

        self.ones = nn.Parameter(
            torch.ones(num_classes, num_prototypes, latent_channels, 1, 1),
            requires_grad=False,
        )

    def forward(self, feats: torch.tensor):
        # For efficiency, compute (x-p)^2 as x2-2xp+p2
        num_classes, num_protos, _, _, _ = self.prototypes.shape
        prototypes = self.prototypes.view(num_classes * num_protos, -1, 1, 1)

        x2 = torch.square(feats)
        x2_patch_sum = nn.functional.conv2d(
            input=x2, weight=self.ones.view(prototypes.shape)
        )

        p2 = torch.square(prototypes)
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = nn.functional.conv2d(input=feats, weight=prototypes)
        dist = nn.functional.relu(x2_patch_sum - 2 * xp + p2_reshape)

        return dist.view(-1, num_classes, num_protos, dist.shape[-2], dist.shape[-1])


@pmf.register_prototype(name="l2")
class L2DSimPrototypeModule(_L2PrototypeModule):
    def __init__(self, num_classes: int, num_prototypes: int, latent_channels: int):
        super().__init__(num_classes, num_prototypes, latent_channels)

    def forward(self, feats: torch.tensor):
        dist = super().forward(feats)
        density = torch.log((dist + 1) / (dist + 1e-6))
        return density


@pmf.register_prototype(name="l2-distance")
class L2DistPrototypeModule(_L2PrototypeModule):
    def __init__(self, num_classes: int, num_prototypes: int, latent_channels: int):
        super().__init__(num_classes, num_prototypes, latent_channels)

    def forward(self, feats: torch.tensor):
        dist = super().forward(feats)
        density = torch.log((dist + 1) / (dist + 1e-6))
        return density, dist


@pmf.register_prototype(name="l2-negated")
class L2PrototypeModule(_L2PrototypeModule):
    def __init__(self, num_classes: int, num_prototypes: int, latent_channels: int):
        super().__init__(num_classes, num_prototypes, latent_channels)

    def forward(self, feats: torch.tensor):
        dist = super().forward(feats)
        density = -dist
        return density


@pmf.register_prototype(name="gaussian")
class GaussianPrototypeModule(L2PrototypeModule):
    def __init__(self, num_classes, num_prototypes, latent_channels):
        super().__init__(num_classes, num_prototypes, latent_channels)

        self.sigmas = nn.Parameter(torch.ones(num_classes, num_prototypes, 1) * 0.5)

    def forward(self, feats):
        # Copmute Gaussian density of feats and prototypes with sigma
        num_classes, num_protos, _, _, _ = self.prototypes.shape
        prototypes = self.prototypes.view(num_classes * num_protos, -1, 1, 1)
        sigmas = self.sigmas.view(num_classes * num_protos, 1, 1, 1)

        x2 = torch.square(feats)
        x2_patch_sum = nn.functional.conv2d(
            input=x2, weight=self.ones.view(prototypes.shape)
        )

        p2 = torch.square(prototypes)
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = nn.functional.conv2d(input=feats, weight=prototypes)

        dist = nn.functional.relu(x2_patch_sum - 2 * xp + p2_reshape)

        sigmas = self.sigmas.view(1, num_classes * num_protos, 1, 1)
        dist = torch.square(dist) / (2 * torch.square(sigmas) + 1e-6)
        density = -dist - torch.log(2 * torch.pi * sigmas)

        return density.view(
            -1, num_classes, num_protos, density.shape[-2], density.shape[-1]
        )
