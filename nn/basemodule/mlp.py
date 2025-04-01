"""多层感知机模块"""

from typing import TYPE_CHECKING

import torch


class MLP(torch.nn.Module):
    """多层感知机模块

    全连接 -> GELU -> 全连接
    """

    if TYPE_CHECKING:
        def norm(self, x: torch.Tensor) -> torch.Tensor:
            """层归一化 (针对单个样本的不同特征进行归一化)

            Args:
                x (torch.Tensor): 输入张量

            Returns:
                torch.Tensor: 归一化后的张量
            """
            ...

        def main(self, x: torch.Tensor) -> torch.Tensor:
            """主网络 (全连接 -> GELU -> 全连接)

            Args:
                x (torch.Tensor): 输入张量, shape:(Batch, sequence Length, Channel)

            Returns:
                torch.Tensor: 输出张量, shape:(Batch, sequence Length, Channel)
            """
            ...

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量, shape:(Batch, sequence Length, Channel)

            Returns:
                torch.Tensor: 输出张量, shape:(Batch, sequence Length, Channel)
            """

    def __init__(self, channels: int, expansion: int):
        """初始化一个两层的多层感知机

        Args:
            channels (int): 输入的维度
            expansion (int): 扩大系数. 隐藏层维度 = 输入维度 * 扩大系数
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))
