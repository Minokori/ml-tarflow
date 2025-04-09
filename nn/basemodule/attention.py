"""注意力模块"""
from typing import TYPE_CHECKING, Literal, overload

import torch


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# H: 注意力头数 (number of attention heads)
# D: 每个注意力头的维度 (dimension of each attention head)
# H * D =C

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
                x (torch.Tensor): 输入张量, shape: (B, L, C)
            Returns:
                torch.Tensor: 在 Channel 维度 `concat`的 QKV 矩阵, shape: (B, L, 3*C)
            """
            ...

        @overload
        def proj(self, x: torch.Tensor) -> torch.Tensor:
            """投影

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 投影后的张量, shape: (B, L, C)
            """
            ...

    def __init__(self, in_channels: int, head_channels: int):
        """初始化

        Args:
            in_channels (int): 输入的维度 C
            head_channels (int): 为每个注意力头分配的维度 D
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
        """注意力头数H"""
        self.sqrt_scale = head_channels ** (-0.25)
        """每个注意力头的缩放点积注意力的缩放因子的平方根"""
        self.sample = False
        """是否为采样(逆运算)模式"""
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        """K矩阵的缓存"""
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        """V矩阵的缓存"""

    def forward_sdpa(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 1.0, which_cache: str = 'cond'
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
        q, k, v = self.qkv(x).reshape(B, L, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, l, d)
        # x -> qkv       shape: (b, l, 3 * c)
        #   -> reshape   shape: (b, l, 3 * h, d), h = num_heads, d = head_dim, h * d = c
        #   -> transpose shape: (b, 3 * h, l, d)
        #   -> chunk     shape: 3 * (b, h, l, d)

        # 逆运算时,计算 attention 时的 mask 为空, 使用缓存的kv矩阵计算
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            # shape: (i+1) * (B,h,1,d)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)
            # shape: (b, h, i+1, d)
            # region NOTE
            # k 缓存和 v 缓存在逆运算之前清空, 逆运算是对输入x(B, L, C) 按行计算
            # 每次逆计算时, 输入的是 x_i (B,1,C), i 是序列元素的索引, 0 <= i < L
            # 计算 x_i 时, k/v 缓存中各存储了 i 个 k/v, 每个 shape: (B,h,1,d)
            # 计算 x_i 后, k,v 缓存中各存储了 i+1 个 k/v
            # 最后一次逆计算时, i= L-1, torch.cat() 后 shape: (B,h,L,d)
            # endregion

        # 计算缩放因子 $$ d_k $$
        scale = self.sqrt_scale**2 / tau
        # region NOTE
        # 该操作在原论文(11)式前描述, 将 attention 的 log 除以 tau
        # endregion

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
        self, x: torch.Tensor, mask: torch.Tensor | None = None, tau: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SDPA:
            return self.forward_sdpa(x, mask, tau, which_cache)
        return self.forward_base(x, mask, tau, which_cache)
