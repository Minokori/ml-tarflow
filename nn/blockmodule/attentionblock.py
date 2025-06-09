"""残差注意力块 (一个注意力机制模块 + 一个全连接层, 串联)"""
from typing import TYPE_CHECKING, Literal

from config import AttentionBlockConfig
import torch

from nn.basemodule import *


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# H: 注意力头数 (number of attention heads)
# C 必须能够整除 H


class AttentionBlock(torch.nn.Module):
    """残差注意力块 (一个注意力机制模块 + 一个全连接层, 串联)

    >>> input += attention(input)
        input += mlp(input)
        return input
    """

    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, tau: float = 1.0,
                     which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量. shape: (B, L, C)
                attn_mask (torch.Tensor | None, optional): 注意力遮罩. Defaults to None.
                tau (float, optional): 手动注入温度项. Defaults to 1.0.
                which_cache (str, optional): 缓存选择(和是否条件引导有关). Defaults to 'cond'.

            Returns:
                torch.Tensor: 输出张量, shape: (B, L, C)
            """
            ...

    def __init__(self, config:AttentionBlockConfig):
        """初始化"""
        self._config = config
        super().__init__()
        self.attention = Attention(self._config.AttentionConfig)
        """注意力块"""
        self.mlp = MLP(self._config.MLPConfig)
        """MLP 块"""
    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, tau: float = 1.0, which_cache: Literal["cond", "uncond"] = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, tau, which_cache)
        x = x + self.mlp(x)
        return x
