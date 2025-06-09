"""多层感知机模块"""

from typing import TYPE_CHECKING, overload

from config import MLPConfig
import torch


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)


class MLP(torch.nn.Module):
    """多层感知机模块

    全连接 -> GELU -> 全连接
    """
    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            """前向传播

            shape: (B, L, C) -> (B, L, C)

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 输出张量, shape: (B, L, C)
            """
            ...

        def main(self, x: torch.Tensor) -> torch.Tensor:
            """主网络 (全连接 -> GELU -> 全连接)

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 输出张量, shape: (B, L, C)
            """
            ...

        def norm(self, x: torch.Tensor) -> torch.Tensor:
            """层归一化 (针对单个样本的不同特征进行归一化)

            Args:
                x (torch.Tensor): 输入张量

            Returns:
                torch.Tensor: 归一化后的张量
            """
            ...

    def __init__(self, config:MLPConfig):
        """初始化一个两层的多层感知机"""
        self._config = config
        super().__init__()
        self.norm = torch.nn.LayerNorm(self._config.channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(self._config.channels, self._config.channels_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(self._config.channels_hidden, self._config.channels),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))
