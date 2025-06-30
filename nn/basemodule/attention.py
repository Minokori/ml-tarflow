"""注意力模块"""
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from config import AttentionConfig


# region 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# H: 注意力头数 (number of attention heads)
# D: 每个注意力头的维度 (dimension of each attention head)
# H * D =C
# endregion

class Attention(torch.nn.Module):
    """注意力模块

    + 相比于 TarFlow 原始论文, 删除了 `forward_base()`, 采用点注意力机制实现
    """


    if TYPE_CHECKING:
        def __call__(self,
                     x: torch.Tensor,
                     mask: torch.Tensor,
                     which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
            """使用点积注意力机制(SDPA)进行注意力计算

            Args:
                x (torch.Tensor): 输入, shape: (B, L, C)
                mask (torch.Tensor): 注意力遮罩
                which_cache (Literal[&quot;cond&quot;, &quot;uncond&quot;], optional): 是否使用条件引导. Defaults to 'cond'.

            Returns:
                attention (torch.Tensor): 注意力张量, shape: (B, L, C)
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
        """K矩阵的缓存. 仅在逆运算时使用

        shape: (B, 1, C) * i, i 为逆运算的子步骤次数
        """
        self._v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}  # type: ignore
        """V矩阵的缓存, 仅在逆运算时使用

        shape: (B, 1, C) * i, i 为逆运算的子步骤次数
        """

        self._k_gsj_cache: dict[str, torch.Tensor] = {'cond': torch.tensor([]), 'uncond': torch.tensor([]), }
        """K矩阵 GSJ 模式的缓存. 仅在逆运算时使用

        shape = (b,h,J,d)
        """
        self._v_gsj_cache: dict[str, torch.Tensor] = {'cond': torch.tensor([]), 'uncond': torch.tensor([]), }
        """V矩阵 GSJ 模式的缓存. 仅在逆运算时使用


        shape = (b, h, J, d)
        """
        self._GSJmode: Literal["GS", "J", "GSJ"] = "J"
        # endregion

    # region properties

    @property
    def GSJmode(self) -> Literal["GS", "J", "GSJ"]:
        """GSJ 模式

        + `"J"` : Jacobi 模式, 正向传播时也是该模式
        + `"GSJ"` : 混合模式
        + `"GS"` : Gauss-Seidel 模式, Tar-Flow的模式
        """
        return self._GSJmode

    @GSJmode.setter
    def GSJmode(self, value: Literal["GS", "J", "GSJ"]):
        """设置 GSJ 模式, 设置的同时会清空缓存"""
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
    # endregion

    def cat_kv_temp(self, which_cache: Literal["cond", "uncond"] = 'cond') -> None:
        """把 gsj 缓存内的 K, V 矩阵添加到一般缓存中

        Args:
            which_cache (Literal[&quot;cond&quot;, &quot;uncond&quot;], optional): _description_. Defaults to 'cond'.
        """
        self._k_cache[which_cache].append(self._k_gsj_cache[which_cache])
        self._v_cache[which_cache].append(self._v_gsj_cache[which_cache])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, which_cache: Literal["cond", "uncond"] = 'cond'
    ) -> torch.Tensor:
        """使用点积注意力机制(SDPA)进行注意力计算

        Args:
            x (torch.Tensor): 输入, shape: (B, L, C)
            mask (torch.Tensor): 注意力遮罩
            which_cache (Literal[&quot;cond&quot;, &quot;uncond&quot;], optional): 是否使用条件引导. Defaults to 'cond'.

        Returns:
            attention (torch.Tensor): 注意力张量, shape: (B, L, C)
        """
        match self.GSJmode:
            case "J":  # 正向运算
                return self._forward(x, mask)
            case "GSJ":  # 逆向运算, GSJ 模式
                return self._reverse_GSJ(x, mask, which_cache)
            case "GS":  # 逆向运算, GS 模式
                return self._reverse_GS(x, which_cache)

    def _forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

        # 计算缩放因子 $$ d_k $$
        # region NOTE
        # 该操作在原论文(11)式前描述, 将 attention 的 log 除以 tau
        # endregion
        scale = self._sqrt_scale**2 / self._config.scale

        # 正向计算时, mask 为下三角矩阵
        mask = mask.bool()

        # 计算注意力权重
        assert q.shape[-2] == mask.shape[-2] and k.shape[-2] == mask.shape[-1], "mask 的形状与 q, k 的形状不匹配."
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)  # shape: (b, h, l, d)

        x = attn.transpose(1, 2).reshape(B, L, C)
        # attn -> transpose shape: (b, l, h, d)
        #      -> reshape   shape: (b, l, c), h * d = c
        x = self.proj(x)  # shape: (b, l, c)
        return x

    def _reverse_GSJ(self, x: torch.Tensor, mask: torch.Tensor,
                     which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
        """逆向运算, GSJ 模式

        Args:
            x (torch.Tensor): 输入张量, (**输入序列的 Jacobi 块**) shape: (B, J, C)
            mask (torch.Tensor | None, optional): 注意力mask. Defaults to None.
            which_cache (str, optional): 缓存模式. Defaults to 'cond'.

        Returns:
            torch.Tensor: 输出张量, shape: (B, J, C)
        """

        B, J, C = x.size()
        x = self.norm(x.float()).type(x.dtype)

        q, k, v = self.qkv(x).reshape(B, J, 3 * self._config.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, J, d)

        # 缓存第 i 个 Jacobi 块的 K, V 矩阵
        self._k_gsj_cache[which_cache] = k  # shape (B, h, J, d)
        self._v_gsj_cache[which_cache] = v  # shape (B, h, J, d)

        # 雅可比块的索引, 也即当前逆运算的子步骤次数
        jacobi_block_idx = len(self._k_cache[which_cache])

        # 拼接历史的 Jacobi 块的 K, V 矩阵, K.shape = V.shape = (B, h, J*(i+1), d)
        if (jacobi_block_idx == 0):
            k = self._k_gsj_cache[which_cache]
            v = self._v_gsj_cache[which_cache]
        else:
            k = torch.cat([torch.cat(self._k_cache[which_cache], dim=2), self._k_gsj_cache[which_cache]], dim=2)
            v = torch.cat([torch.cat(self._v_cache[which_cache], dim=2), self._v_gsj_cache[which_cache]], dim=2)
            # torch.cat 内的 第一个 torch.cat 结果: shape  = (B, h, J*i, d)
            # torch.cat 内的 第二部分 shape = (B, h, J, d)
            # torch.cat 的结果: (B, h, J*(i+1), d)

        scale = self._sqrt_scale**2 / self._config.scale

        # mask.shape = (      J, J*(i+1)   )
        #    q.shape = (b, h, J,          d)
        # k, v.shape = (b, h,    J*(i+1), d)
        if mask is not None:
            assert q.shape[-2] == mask.shape[-2] and k.shape[-2] == mask.shape[-1] , "mask 的形状与 q, k 的形状不匹配."
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)  # shape = (B,h,J,d)

        x = attn.transpose(1, 2).reshape(B, J, C)
        x = self.proj(x)  # shape: (b, J, c)
        return x

    def _reverse_GS(self, x: torch.Tensor, which_cache: Literal["cond", "uncond"] = 'cond') -> torch.Tensor:
        """逆向运算, GS 模式 (TarFlow 论文中的逐行逆向运算)

        Args:
            x (torch.Tensor): 输入张量的 **(单行,L=1)**, shape: (B, 1, C)
            which_cache (Literal[&quot;cond&quot;, &quot;uncond&quot;], optional): 是否有条件引导. Defaults to 'cond'.

        Returns:
            x (torch.Tensor): 输出张量, shape: (B, 1, C)
        """
        B, L, C = x.size()
        assert L == 1, "x 没有以单行输入"
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, L, 3 * self._config.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, 1, d)

        self._k_cache[which_cache].append(k)  # shape: (b, h, 1, d) * i -> (b, h, 1, d) * (i + 1)
        self._v_cache[which_cache].append(v)  # shape: (b, h, 1, d) * i -> (b, h, 1, d) * (i + 1)
        k = torch.cat(self._k_cache[which_cache], dim=2)  # shape = (b, h, 1, d) * (i+1) -> (b, h, i+1, d)
        v = torch.cat(self._v_cache[which_cache], dim=2)  # shape = (b, h, 1, d) * (i+1) -> (b, h, i+1, d)

        # k,v.shape = (b,h,i+1,d)
        scale = self._sqrt_scale**2 / self._config.scale
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)  # (b,h,1,d)

        x = attn.transpose(1, 2).reshape(B, L, C)  # shape: (b,1,c)
        x = self.proj(x)  # shape: (b, 1, c)
        return x
