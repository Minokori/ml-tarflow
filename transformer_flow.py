#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import TYPE_CHECKING, overload

import torch

from nn.basemodule import *
from nn.blockmodule import *
from nn.metablock import MetaBlock


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor
    """先验分布.

    `NVP` 模式下的 全为1的矩阵, 但 `VP` 模式下是可学习的参数

    `shape:(L,C * P * P)`
    """

    if TYPE_CHECKING:
        @overload
        def __call__(self,
                     x: torch.Tensor,
                     y: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量(图片), shape: (B, C, W, W)
                y (torch.Tensor | None, optional): 输入张量的标签, shape: (B). Defaults to None.

            Returns:
                tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: 输出 (B, L, C*P*P), 每层的输出 , logdets (B,)
            """
            ...

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
        """初始化模型

        Args:
            in_channels (int): 输入张量(图片)的通道数, 记为`C`
            img_size (int): 输入图像的边长, 记为 `W`
            patch_size (int): 图像分块的边长, 记为 `P`
            channels (int): MetaBlock 中的隐藏层通道数, 记为 `C_hidden`
            num_blocks (int): MetaBlock 的数量
            layers_per_block (int): MetaBlock 中的层数
            nvp (bool, optional): 是否使用 `NVP` 模式. Defaults to True.
            num_classes (int, optional): 分类数量, 用于引导网络训练. Defaults to 0.
        """
        super().__init__()
        self.img_size = img_size
        """输入图像的边长, 记为 `W`"""
        self.patch_size = patch_size
        """图像块的边长,记为 `P`"""
        self.num_patches = (img_size // patch_size) ** 2
        """图像被分割成的块数,记为L

        由于每一块被当成序列的一个元素, 故使用 `L` 表示, L = (W/P)^2
        """
        permutations: list[Permutation] = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]
        """置换块, 一个单位置换块,一个翻转置换块"""

        # 初始化 MetaBlock 块
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
        """Meta Block"""
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""将输入的图片张量转为块序列张量

        $$ Sequence_{(B, L, CP^2)} = patchify(Img_{(B,C,W,W)}), L = (\frac{W}{P})^2$$

        Args:
            x (torch.Tensor): 图片张量, shape: (B, C, W, W)

        Returns:
            torch.Tensor: 块序列张量, shape: (B, L, C*P*P)
        """
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)  # shape: (B, C_img, L)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""将输入的块序列张量转为图片张量

        $$ Img_{(B,C,W,W)} = fold(Sequence_{(B, L, CP^2)}), L = (\frac{W}{P})^2$$

        Args:
            x (torch.Tensor):块序列张量, shape: (B, L, C*P*P)

        Returns:
            torch.Tensor: 图片张量, shape: (B, C, W, W)
        """
        u = x.transpose(1, 2)  # shape: (B, C*P*P, L)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量(图片), shape: (B, C, W, W)
            y (torch.Tensor | None, optional): logdet. Defaults to None.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]: 输出 (B, L, C*P*P), 每层的输出 , logdets (B,)
        """
        x = self.patchify(x)  # shape: (B, L, C*P*P)

        # 初始化每一层的输出
        outputs: list[torch.Tensor] = []

        # 初始化 雅可比行列式的 log 值
        logdets = torch.zeros((), device=x.device)  # shape: ()

        # 计算并保留每层的输出
        for block in self.blocks:
            x, logdet = block(x, y)  # shape: (B, L, C), (B)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        """        用 z^2 和 var 的插值更新

        $$ var = var +  (z^2 - var) \times var_{LR} $$


        Args:
            z (torch.Tensor): var 的优化目标
        """
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        """计算模型最终输出和其logdets的loss, 用作损失函数

        Args:
            z (torch.Tensor): 模型的最终输出, shape:(B,L,C)
            logdets (torch.Tensor): 模型每层Flow 的雅各比行列式值的乘积的log值, shape:(B)
        Returns:
            torch.Tensor: 训练损失
        """
        return 0.5 * z.pow(2).mean() - logdets.mean()
        # region NOTE 损失函数
        # 原论文公式(6):
        #
        # $$ \hspace{5em}loss = min_f 0.5||z^T||^2_2 + \sum^{T-1}_{t=0}\sum^{N-1}_{n=0}\sum^{D-1}_{j=0} \alpha_i^t(z^t_{<i})_j $$
        #
        #
        # 原论文公式(5):
        #
        # $$ \hspace{4em}logdet^t = -\sum^{N-1}_{n=0}\sum^{D-1}_{j=0} \alpha_i^t(z^t_{<i})_j $$
        #
        # 即第 t 层 MetaBlock 的 logdet 值即为(6)式的后半部分, 代入(6)式, 得:
        #
        # $$\hspace{4em}loss = min_f0.5||z^T||^2_2 - \sum^{T-1}_{t=0} logdet^t$$
        #
        # 其中 $$ ||z^T||^2_2 $$表示向量 $$z^T$$ 的 L2 范数的平方, 即向量中每个元素的平方和, 即:
        # z.pow(2).sum(dim =[1,2])
        # 则第i个样本的损失为:
        # 0.5 * z[i].pow(2).sum(dim=[1,2]) - logdets[i]
        # 所有样本的损失的平均值为:
        # 0.5 * z.pow(2).sum(dim=[1,2]).mean() - logdets.mean()
        # sum() 运算在这里显然可以省略, 即为 return 语句的表达式:
        # 0.5 * z.pow(2).mean() - logdets.mean()
        # endregion

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
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor | None, optional): _description_. Defaults to None.
            guidance (float, optional): _description_. Defaults to 0.
            guide_what (str, optional): _description_. Defaults to 'ab'.
            attn_temp (float, optional): _description_. Defaults to 1.0.
            annealed_guidance (bool, optional): _description_. Defaults to False.
            return_sequence (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor | list[torch.Tensor]: _description_
        """
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
