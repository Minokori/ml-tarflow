#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch

from nn.basemodule import *
from nn.blockmodule import *
from nn.metablock import MetaBlock


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks: list[MetaBlock] = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x)

        # 初始化每一层的输出
        outputs: list[torch.Tensor] = []

        # 初始化 雅可比行列式的 log 值
        logdets = torch.zeros((), device=x.device)

        # 计算并保留值
        for block in self.blocks:
            x, logdet = block(x, y)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)

        if not return_sequence:
            return x
        else:
            return seq
