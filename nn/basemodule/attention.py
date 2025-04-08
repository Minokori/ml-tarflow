"""注意力模块"""
from typing import TYPE_CHECKING, Literal, overload

import torch


class Attention(torch.nn.Module):
    """注意力模块
    """
    USE_SDPA: bool = True
    """是否使用 `Scaled Dot-Product Attention(SDPA)`, 点积注意力机制"""

    if TYPE_CHECKING:
        @overload
        def __call__(self,
                     x: torch.Tensor,
                     mask: torch.Tensor | None = None,
                     temp: float = 1.0,
                     which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
            """计算注意力

                `(b, l, c) -> (b, l, c)`

            Args:
                x (torch.Tensor): 输入张量 shape = (Batch, sequence Length, Channel)
                mask (torch.Tensor | None, optional): 计算注意力时的遮罩. Defaults to None.
                temp (float, optional): _description_. Defaults to 1.0.
                which_cache (str, optional): 有无条件指导. Defaults to 'cond'.

            Returns:
                torch.Tensor: 输出 ,shape = input shape (B, L, C)
            """
            ...

        @overload
        def norm(self, x: torch.Tensor) -> torch.Tensor:
            """层归一化(针对单个样本的不同特征进行归一化)

            Args:
                x (torch.Tensor): 输入张量

            Returns:
                torch.Tensor: 归一化后的张量
            """
            ...

        @overload
        def qkv(self, x: torch.Tensor) -> torch.Tensor:
            """计算 Query, Key, Value 的权重矩阵

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channels)

            Returns:
                torch.Tensor: 在 Channel 维度 `concat`的 QKV 矩阵, shape: (Batch, sequence Length, 3 * Channels)
            """
            ...

        @overload
        def proj(self, x: torch.Tensor) -> torch.Tensor:
            """投影

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channels)

            Returns:
                torch.Tensor: 投影后的张量, shape: (Batch, sequence Length, Channels)
            """
            ...

    def __init__(self, in_channels: int, head_channels: int):
        """初始化

        Args:
            in_channels (int): 输入的维度
            head_channels (int): 为每个注意力头分配的维度
        """
        assert in_channels % head_channels == 0  # 确保输入维度能被注意力头数整除
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        """层归一化(针对单个样本的不同特征进行归一化)"""
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        """Query, Key, Value 矩阵的权重"""
        self.proj = torch.nn.Linear(in_channels, in_channels)
        """投影"""
        self.num_heads = in_channels // head_channels
        """注意力头数"""
        self.sqrt_scale = head_channels ** (-0.25)
        """每个注意力头的 缩放点积注意力的缩放因子的平方根"""
        self.sample = False
        """是否为采样(逆运算)模式"""
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        """K矩阵的缓存"""
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        """V矩阵的缓存"""

    def forward_sdpa(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        """使用点积注意力机制(SDPA)进行前向传播

        Args:
            x (torch.Tensor): 输入, shape: (Batch, Sequence Length, Channels)
            mask (torch.Tensor | None, optional): _description_. Defaults to None.
            temp (float, optional): _description_. Defaults to 1.0.
            which_cache (str, optional): _description_. Defaults to 'cond'.

        Returns:
            torch.Tensor: _description_
        """

        # 获取输入的批次大小、序列长度和通道数
        B, L, C = x.size()

        # 进行层归一化
        x = self.norm(x.float()).type(x.dtype)

        # 计算 Q, K, V 矩阵
        q, k, v = self.qkv(x).reshape(B, L, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)
        # x -> qkv       shape: (b, l, 3 * c)
        #   -> reshape   shape: (b, l, 3 * h, d), h = num_heads, d = head_dim, h * d = c
        #   -> transpose shape: (b, 3 * h, l, d)
        #   -> chunk     shape: 3 * (b, h, l, d)

        # region TODO 和逆运算相关
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # shape: (b, h,2*t, d)
            v = torch.cat(self.v_cache[which_cache], dim=2)
        # endregion

        # 计算缩放因子 $$ d_k $$
        scale = self.sqrt_scale**2 / temp

        # 是否执行 attention mask 操作
        if mask is not None:
            mask = mask.bool()

        # 计算注意力权重
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)  # attn shape: (b, h, l, d)

        x = attn.transpose(1, 2).reshape(B, L, C)
        # attn -> transpose shape: (b, l, h, d)
        #      -> reshape   shape: (b, l, c), h * d = c
        x = self.proj(x)  # shape: (b, l, c)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SDPA:
            return self.forward_sdpa(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)
