"""TarFlow模型"""
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import TYPE_CHECKING

import torch

from config import TarflowConfig
from nn import ReverseHyperParameters
from nn.basemodule import *
from nn.blockmodule import *
from nn.metablock import MetaBlock


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# P: 图像块的边长 (patch size)
# W: 图像的边长 (image size)
# L = W * W /P /P
# H: 注意力头数 (number of attention heads)
# D: 每个注意力头的维度 (dimension of each attention head)
# H * D =C
# C_hidden: 隐藏层通道数

class TarFlow(torch.nn.Module):
    """TarFlow模型"""

    if TYPE_CHECKING:
        def __call__(self,
                     x: torch.Tensor,
                     y: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量(图片), shape: (B, C, W, W)
                y (torch.Tensor | None, optional): 输入张量的标签, shape: (B). Defaults to None.

            Returns:
                output,perlayeroutput,logdets (tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]): 输出 (B, L, C*P*P), 每层的输出 , logdets (B,)
            """
            ...

        @property
        def var(self) -> torch.Tensor:
            """先验分布

            `NVP` 模式下的 全为1的矩阵, 但 `VP` 模式下是可学习的参数

            `shape:(L,C * P * P)`
            """
            ...

    def __init__(self, config: TarflowConfig):
        super().__init__()
        self._config = config
        self.permutations: list[Permutation] = [PermutationIdentity(config.num_patches), PermutationFlip(config.num_patches)]
        blocks = []
        for i in range(config.blocks_num):
            blocks.append(
                MetaBlock(
                    config.MetaBlockConfig,
                    permutation=self.permutations[i % 2],
                )
            )
        self.blocks: list[MetaBlock] = torch.nn.ModuleList(blocks)  # type: ignore
        self.register_buffer('var', torch.ones(self._config.num_patches, self._config.channels_patched))
        pass

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        r"""将输入的图片张量转为块序列张量

        $$ Sequence_{(B, L, CP^2)} = patchify(Img_{(B,C,W,W)}), L = (\frac{W}{P})^2$$

        Args:
            x (torch.Tensor): 图片张量, shape: (B, C, W, W)

        Returns:
            torch.Tensor: 块序列张量, shape: (B, L, C*P*P)
        """
        u = torch.nn.functional.unfold(x, self._config.patch_size, stride=self._config.patch_size)  # shape: (B, C_img, L)
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
        return torch.nn.functional.fold(u, (self._config.img_size, self._config.img_size), self._config.patch_size, stride=self._config.patch_size)

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
            x, logdet,d = block(x, y)  # shape: (B, L, C), (B)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        """计算模型最终输出和其logdets的loss, 用作损失函数

        Args:
            z (torch.Tensor): 模型的最终输出, shape:(B,L,C)
            logdets (torch.Tensor): 模型每层Flow 的雅各比行列式值的乘积的log值, shape:(B)
        Returns:
            torch.Tensor: 训练损失
        """
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
        hyper_para: ReverseHyperParameters = ReverseHyperParameters(),
        num_GS_list: list[int] | None = None,
        max_jacobi_list: list[int] | None = None,
        guess_list: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor): 输入张量(序列), shape: (B, L, C*P*P)
            y (torch.Tensor | None, optional): _description_. Defaults to None.
            guidance (float, optional): _description_. Defaults to 0.
            guide_what (str, optional): _description_. Defaults to 'ab'.
            attn_temp (float, optional): _description_. Defaults to 1.0.
            annealed_guidance (bool, optional): _description_. Defaults to False.
            return_sequence (bool, optional): 是否返回序列. 若为否, 则仅返回最终结果(图片); 否则, 将返回每层 MetaBlock 处理后的结果 Defaults to False.

        Returns:
            torch.Tensor | list[torch.Tensor]: _description_
        """

        num_GS_list = num_GS_list or [0] * self._config.blocks_num
        max_jacobi_list = max_jacobi_list or [0] * self._config.blocks_num
        guess_list = guess_list or [0] * self._config.blocks_num

        x = x * self.var.sqrt()

        for block_idx, block in enumerate(reversed(self.blocks)):

            if block_idx == len(self.blocks) - 1:
                hyper_para.incre1 = True
            else:
                hyper_para.incre1 = False

            hyper_para.num_GS = num_GS_list[block_idx]
            hyper_para.max_jacobi = max_jacobi_list[block_idx]
            hyper_para.zero_guess = guess_list[block_idx]

            x = block.reverse(x, y, hyper_para)


        x = self.unpatchify(x)

        return x
