"""残差注意力块 (一个注意力机制模块 + 一个全连接层, 串联"""
from typing import TYPE_CHECKING

import torch

from nn.basemodule import *


class AttentionBlock(torch.nn.Module):
    """残差注意力块 (一个注意力机制模块 + 一个全连接层, 串联)

    >>> input += attention(input)
        input += mlp(input)
        return input
    """

    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond') -> torch.Tensor:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量. shape:(Batch, sequence Length, Channel)
                attn_mask (torch.Tensor | None, optional): _description_. Defaults to None.
                attn_temp (float, optional): _description_. Defaults to 1.0.
                which_cache (str, optional): _description_. Defaults to 'cond'.

            Returns:
                torch.Tensor: 输出张量, shape = input shape(B, L, C)
            """

    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x
