"""注意力模块"""
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch
from torch.linalg import vector_norm

from config import AttentionConfig


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# H: 注意力头数 (number of attention heads)
# D: 每个注意力头的维度 (dimension of each attention head)
# H * D =C
# [TODO] sample 删掉, 换成GSJmode, J是正向, GS/GSJ 是逆向
class Attention(torch.nn.Module):
    """注意力模块

    + 相比于 TarFlow 原始论文, 删除了 `forward_base()`, 默认采用点注意力机制实现
    """


    if TYPE_CHECKING:
        def __call__(self,
                     x: torch.Tensor,
                     mask: torch.Tensor | None = None,
                     tau: float = 1.0,
                     which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
            """计算注意力

                shape: (B, L, C) -> (B, L, C)

            Args:
                x (torch.Tensor): 输入张量, shape = (B, L, C)
                mask (torch.Tensor | None, optional): 计算注意力时的遮罩. Defaults to None.
                tau (float, optional): 手动注入温度项. Defaults to 1.0.
                which_cache (str, optional): 有无条件指导. Defaults to 'cond'.

            Returns:
                torch.Tensor: 输出张量 ,shape: (B, L, C)
            """
            ...

        def norm(self, x: torch.Tensor) -> torch.Tensor:
            """层归一化(针对单个样本的不同特征进行归一化)

            Args:
                x (torch.Tensor): 输入张量

            Returns:
                torch.Tensor: 归一化后的张量
            """
            ...

        def qkv(self, x: torch.Tensor) -> torch.Tensor:
            """计算 Query, Key, Value 的权重矩阵

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)
            Returns:
                torch.Tensor: 在 Channel 维度 `concat`的 QKV 矩阵, shape: (B, L, 3*C)
            """
            ...

        def proj(self, x: torch.Tensor) -> torch.Tensor:
            """投影

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 投影后的张量, shape: (B, L, C)
            """
            ...

    def __init__(self, config:AttentionConfig):
        """初始化

        Args:
            channels_in (int): 输入的维度 C
            channels_per_head (int): 每个注意力头分配的维度 D
        """
        self._config = config
        super().__init__()
        self.norm = torch.nn.LayerNorm(self._config.channels_in)
        """层归一化(针对单个样本的不同特征进行归一化)"""
        self.qkv = torch.nn.Linear(self._config.channels_in, self._config.channels_in * 3)
        """Query, Key, Value 矩阵的权重"""
        self.proj = torch.nn.Linear(self._config.channels_in, self._config.channels_in)
        """投影"""

        # region 缓存 K, V 矩阵
        self._k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}  # type: ignore
        """K矩阵的缓存. 仅在逆运算时使用"""
        self._v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}  # type: ignore
        """V矩阵的缓存, 仅在逆运算时使用"""

        self._k_gsj_cache: dict[str, torch.Tensor] = {'cond': torch.tensor([]), 'uncond': torch.tensor([]), }
        """K矩阵 GSJ 模式的缓存. 仅在逆运算时使用"""
        self._v_gsj_cache: dict[str, torch.Tensor] = {'cond': torch.tensor([]), 'uncond': torch.tensor([]), }
        """V矩阵 GSJ 模式的缓存. 仅在逆运算时使用"""
        self._GSJmode: Literal["GS", "J", "GSJ"] = "J"
        # endregion

    @property
    def GSJmode(self) -> Literal["GS", "J", "GSJ"]:
        """GSJ 模式

        + `"J"` : 正向运算
        + `"GSJ"` :
        """
        return self._GSJmode

    @GSJmode.setter
    def GSJmode(self, value: Literal["GS", "J", "GSJ"]):
        """设置 GSJ 模式"""
        if value not in ["GS", "J", "GSJ"]:
            raise ValueError("GSJmode must be one of 'GS', 'J', or 'GSJ'")
        self._GSJmode = value
        # 清空缓存
        self._k_cache = {'cond': [], 'uncond': []}
        self._v_cache = {'cond': [], 'uncond': []}
        self._k_gsj_cache = {'cond': torch.tensor([]), 'uncond': torch.tensor([])}
        self._v_gsj_cache = {'cond': torch.tensor([]), 'uncond': torch.tensor([])}

    @cached_property
    def _sqrt_scale(self) -> float:
        """计算每个注意力头的缩放点积注意力的缩放因子的平方根"""
        return self._config.channels_per_head ** (-0.25)

    def _forward_sdpa(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 1.0, which_cache: Literal["cond", "uncond"] = 'cond'
    ) -> torch.Tensor:
        """使用点积注意力机制(SDPA)进行前向传播

        Args:
            x (torch.Tensor): 输入, shape: (B, L, C)
            mask (torch.Tensor | None, optional): 注意力mask. Defaults to None.
            tau (float, optional): 手动注入温度项. Defaults to 1.0.
            which_cache (str, optional): 缓存模式. Defaults to 'cond'.

        Returns:
            torch.Tensor: _description_
        """

        # 获取输入的批次大小、序列长度和通道数
        B, L, C = x.size()

        # 进行层归一化
        x = self.norm(x.float()).type(x.dtype)

        # 计算 Q, K, V 矩阵
        # x -> qkv       shape: (b, l, 3 * c)
        #   -> reshape   shape: (b, l, 3 * h, d), h = num_heads, d = head_dim, h * d = c
        #   -> transpose shape: (b, 3 * h, l, d)
        #   -> chunk     shape: 3 * (b, h, l, d)
        q, k, v = self.qkv(x).reshape(B, L, 3 * self._config.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, l, d)

        # 逆运算时才使用
        match self.GSJmode:
            case "GSJ":  # 逆运算, 每次 forward 传入的  # GSJ 模式传入的mask shape = (J,(i+1)J )

                # len =
                self._k_gsj_cache[which_cache] = k  # shape (B, h, J, d)
                self._v_gsj_cache[which_cache] = v  # shape (B, h, J, d)

                jacobi_block_idx = len(self._k_cache[which_cache])

                if (jacobi_block_idx == 0):  # 索引为 0 的 Jacobi块 的输入, kv shape = (B, h, J, d)
                    k = self._k_gsj_cache[which_cache]
                    v = self._v_gsj_cache[which_cache]
                else:
                    k = torch.cat([torch.cat(self._k_cache[which_cache], dim=2), self._k_gsj_cache[which_cache]], dim=2)
                    v = torch.cat([torch.cat(self._v_cache[which_cache], dim=2), self._v_gsj_cache[which_cache]], dim=2)
                    # torch.cat 内的 第一个 torch.cat 结果: shape  = (B, h, J*i, d)
                    # torch.cat 内的 第二部分 shape = (B, h, J, d)
                    # torch.cat 的结果: (B, h, J*(i+1), d)

            case "GS":
                self._k_cache[which_cache].append(k)
                self._v_cache[which_cache].append(v)
                k = torch.cat(self._k_cache[which_cache], dim=2)
                v = torch.cat(self._v_cache[which_cache], dim=2)
            case _:
                pass


        # 计算缩放因子 $$ d_k $$
        # region NOTE
        # 该操作在原论文(11)式前描述, 将 attention 的 log 除以 tau
        # endregion
        scale = self._sqrt_scale**2 / tau


        # 正向计算时, mask为下三角矩阵, 逆运算时, mask 为 None
        if mask is not None:
            mask = mask.bool()

        # 计算注意力权重
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)  # shape: (b, h, l, d)
        # 逆运算时, 输入为 x_i 时, shape: (B, h, i+1, d)

        x = attn.transpose(1, 2).reshape(B, L, C)
        # attn -> transpose shape: (b, l, h, d)
        #      -> reshape   shape: (b, l, c), h * d = c
        # 逆运算时, 输入为 x_i 时, shape: (B, i+1, C)
        x = self.proj(x)  # shape: (b, l, c)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 1.0, which_cache: Literal["cond", "uncond"] = 'cond'
    ) -> torch.Tensor:
        return self._forward_sdpa(x, mask, tau, which_cache)
