import math

import torch
from torch import nn

from . import pm_factory as pmf


@pmf.register_prototype(name="cosine")
class CosinePrototypeModule(pmf.PrototypeModule):
    def __init__(self, num_classes, num_prototypes, latent_channels):
        super().__init__(num_classes, num_prototypes, latent_channels)
        nn.init.kaiming_normal_(self.prototypes)

    def forward(self, feats):
        # Compute dot product between feats and prototypes
        num_classes, num_protos, _, _, _ = self.prototypes.shape
        prototypes = self.prototypes.view(num_classes * num_protos, -1)

        feats_norm = nn.functional.normalize(feats, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(prototypes, p=2, dim=1)

        dist = nn.functional.conv2d(
            feats_norm, prototypes_norm.view(num_classes * num_protos, -1, 1, 1)
        )
        return dist.view(-1, num_classes, num_protos, dist.shape[-2], dist.shape[-1])


@pmf.register_prototype(name="hyperpg")
class HyperPgPrototypeModule(CosinePrototypeModule):
    def __init__(self, num_classes, num_prototypes, latent_channels):
        super().__init__(num_classes, num_prototypes, latent_channels)
        nn.init.kaiming_normal_(self.prototypes)

        self.mus = nn.Parameter(torch.rand(num_classes, num_prototypes, 1))
        self.sigmas = nn.Parameter(torch.ones(num_classes, num_prototypes, 1) * 0.5)

    def forward(self, feats):
        num_classes, num_protos, _, _, _ = self.prototypes.shape

        prototypes = self.prototypes.view(num_classes * num_protos, -1)

        feats_norm = nn.functional.normalize(feats, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(prototypes, p=2, dim=1)

        dist = nn.functional.conv2d(
            feats_norm, prototypes_norm.view(num_classes * num_protos, -1, 1, 1)
        )

        mus = self.mus.view(1, num_classes * num_protos, 1, 1)
        sigmas = self.sigmas.view(1, num_classes * num_protos, 1, 1)

        dist = torch.square(dist - mus) / (2 * torch.square(sigmas) + 1e-6)
        density = -dist - torch.log(2 * torch.pi * sigmas)

        return density.view(
            -1, num_classes, num_protos, density.shape[-2], density.shape[-1]
        )


@pmf.register_prototype(name="vmf")
class VmfPrototypeModule(CosinePrototypeModule):
    def __init__(self, num_classes, num_prototypes, latent_channels):
        super().__init__(num_classes, num_prototypes, latent_channels)
        nn.init.kaiming_normal_(self.prototypes)

        self.sigmas = nn.Parameter(torch.ones(num_classes, num_prototypes, 1) * 0.5)

    def forward(self, feats):
        num_classes, num_protos, _, _, _ = self.prototypes.shape

        prototypes = self.prototypes.view(num_classes * num_protos, -1)

        feats_norm = nn.functional.normalize(feats, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(prototypes, p=2, dim=1)

        dist = nn.functional.conv2d(
            feats_norm, prototypes_norm.view(num_classes * num_protos, -1, 1, 1)
        )

        sigmas = self.sigmas.view(1, num_classes * num_protos, 1, 1)
        density = dist / sigmas

        return density.view(
            -1, num_classes, num_protos, density.shape[-2], density.shape[-1]
        )


@pmf.register_prototype(name="attention")
class AttentionPrototypeModule(CosinePrototypeModule):
    def forward(self, feats):
        # Compute dot product between feats and prototypes
        num_classes, num_protos, channels, _, _ = self.prototypes.shape
        prototypes = self.prototypes.view(num_classes * num_protos, -1)

        # feats_norm = nn.functional.normalize(feats, p=2, dim=1)
        # prototypes_norm = nn.functional.normalize(prototypes, p=2, dim=1)

        dist = nn.functional.conv2d(
            feats, prototypes.view(num_classes * num_protos, -1, 1, 1)
        )

        dist = dist / math.sqrt(channels)

        return dist.view(-1, num_classes, num_protos, dist.shape[-2], dist.shape[-1])


@pmf.register_prototype(name="gaussian_dot")
class GaussianDotPrototypeModule(AttentionPrototypeModule):
    def __init__(self, num_classes, num_prototypes, latent_channels):
        super().__init__(num_classes, num_prototypes, latent_channels)
        nn.init.kaiming_normal_(self.prototypes)

        self.mus = nn.Parameter(torch.rand(num_classes, num_prototypes, 1))
        self.sigmas = nn.Parameter(torch.ones(num_classes, num_prototypes, 1) * 0.5)

    def forward(self, feats):
        num_classes, num_protos, _, _, _ = self.prototypes.shape

        prototypes = self.prototypes.view(num_classes * num_protos, -1)

        # feats_norm = nn.functional.normalize(feats, p=2, dim=1)
        # prototypes_norm = nn.functional.normalize(prototypes, p=2, dim=1)

        dist = nn.functional.conv2d(
            feats, prototypes.view(num_classes * num_protos, -1, 1, 1)
        )

        mus = self.mus.view(1, num_classes * num_protos, 1, 1)
        sigmas = self.sigmas.view(1, num_classes * num_protos, 1, 1)

        dist = torch.square(dist - mus) / (2 * torch.square(sigmas) + 1e-6)
        density = -dist - torch.log(2 * torch.pi * sigmas)

        return density.view(
            -1, num_classes, num_protos, density.shape[-2], density.shape[-1]
        )
