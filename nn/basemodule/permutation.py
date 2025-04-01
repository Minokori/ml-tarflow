"""置换模块.

用于在每个耦合层的 `h(·)` 函数前将上一层的输出进行打乱(mask),
以便不同特征均能参与到耦合层的计算.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch


class Permutation(ABC, torch.nn.Module):
    """置换模块的基类(抽象类)

        该模块用于置换输入张量 (mask操作),
        继承该类请重载 `forward()` 方法
    """

    def __call__(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        r"""对输入张量 x 进行置换操作

        $$ Output_{(B,L,C)} = Permutation(Input_{(B,L,C)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel)
                dim (int, optional): 要进行置换操作的维度. Defaults to 1.
                inverse (bool, optional): 是否为逆运算. Defaults to False.

            Returns:
                torch.Tensor: 置换后的张量, shape: (Batch, sequence Length, Channel)
            """
        ...

    def __init__(self, seq_length: int):
        """初始化

        Args:
            seq_length (int): 输入张量的序列长度
        """
        super().__init__()
        self.seq_length = seq_length
        """序列长度"""

    @abstractmethod
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    r"""单位置换 (输出 = 输入×单位矩阵), 输出与输入相同

    $$ output_{(B,L,C)} = input_{(B,L,C)} $$
    """

    if TYPE_CHECKING:
        def __call__(self, x, dim: int = 1, inverse: bool = False) -> torch.Tensor:
            r"""对输入张量不做任何操作.

            $$ output_{(B,L,W)} = input_{(B,L,W)} $$

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel)
                dim (int, optional): 要进行置换操作的维度. Defaults to 1.
                inverse (bool, optional): 是否为逆运算. Defaults to False.

            Returns:
                torch.Tensor: 置换后的张量, shape: (Batch, sequence Length, Channel)
            """
            ...

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    r"""翻转置换 (将输入在某一维度上进行翻转)

    $$ output_{(B,[1:l],C)} = input_{(B,[l:1],C)} $$
    """
    if TYPE_CHECKING:
        def __call__(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
            """将输入张量 x 维度 dim 内的元素进行翻转

            $$ output_{(B,[1:l],C)} = input_{(B,[l:1],C)} $$

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel)
                dim (int, optional): 要反转的维度. Defaults to 1.
                inverse (bool, optional): 是否为逆运算. Defaults to False.

            Returns:
                torch.Tensor: 反转后的张量, shape: (Batch, sequence Length, Channel)
            """
            ...

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])
