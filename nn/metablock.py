""""""
from typing import TYPE_CHECKING

import torch

from nn.basemodule import *
from nn.blockmodule import *


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor
    """注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息

    `shape: (Patch, Patch)`
    """

    if TYPE_CHECKING:
        def proj_in(self, x: torch.Tensor) -> torch.Tensor:
            """线性投影层(in).

            将输入的通道映射到计算注意力时的通道

            $$ Output_{(B,L,C-hidden)} = proj_{in}(Input_{(B,L,C-in)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel_x)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (Batch, sequence Length, Channel_hidden)
            """
            ...

        def proj_out(self, x: torch.Tensor) -> torch.Tensor:
            """线性投影层(out).

            将计算注意力时的通道映射到输出的通道, 输出的通道数为 `in_channels * (1 + nvp)`

            $$ Output_{(B,L,C-out)} = proj_{out}(Input_{(B,L,C-hidden)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel_hidden)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (Batch, sequence Length, Channel_out)
            """
            ...
        pass

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        """初始化

        Args:
            in_channels (int): 输入张量的通道数
            channels (int): 隐藏层通道数
            num_patches (int): 图像被分割成的块数
            permutation (Permutation): 置换类型
            num_layers (int, optional): 注意力块数. Defaults to 1.
            head_dim (int, optional): 每个注意力头分配的通道数. Defaults to 64.
            expansion (int, optional): MLP的隐藏层扩大系数. Defaults to 4.
            nvp (bool, optional): 是否使用 `RealNVP` 模式. Defaults to True.
            num_classes (int, optional): 样本类别数. Defaults to 0.
        """
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        """位置嵌入编码使用的矩阵(可学习参数).

        `shape: (B, C_hidden)` """

        self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2) if num_classes else None
        """类别嵌入编码使用的矩阵(可学习参数).

        初始化时, 若`num_classes > 0`, 该参数用于存储每个类别的嵌入编码. 否则为 `None`.

        `shape: (num_classes, 1, C_hidden)`
        """
        self.attn_blocks: list[AttentionBlock] = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion)
             for _ in range(num_layers)])
        self.nvp = nvp
        """是否使用 `NVP (non-volume preserving, 非体积保持)` 模式.

        非体积保持启用时, logdet 不为1
        """

        self.proj_out = torch.nn.Linear(channels, in_channels * (1 + nvp))
        self.proj_out.weight.data.fill_(0.0)

        self.permutation: Permutation = permutation
        """置换操作块."""

        # 注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息,
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量, shape: (Batch, sequence Length, Channel)
            y (torch.Tensor | None, optional): 前一层传来的雅可比行列式的 log 值, 在 `FLOW` 模型中常用 `"logdet"` 作为变量名. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 输出张量, shape: (Batch, sequence Length, Channel), 本层雅可比行列式的 log 值
        """
        # 置换操作, 在原论文内由 $$ \pi(z) $$ 表示
        x = self.permutation(x)  # shape: (B, L, C_x)

        # 缓存x备用
        x_hat = x  # shape: (B, L, C_x)

        # 位置嵌入编码
        pos_embed = self.permutation(self.pos_embed, dim=0)  # shape: (B, C_hidden)
        x = self.proj_in(x) + pos_embed  # shape: (B,L,C_hidden)

        # region TODO 类别嵌入编码? y 是 logdet
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)
        # endregion

        # 计算注意力
        for block in self.attn_blocks:
            x = block(x, self.attn_mask)  # shape: (B, L, C_hidden)
            # region NOTE ⚠ attn_mask 的作用 ⚠
            # 实现仿射耦合层的多块划分, 下式为原论文的公式(3):( $$\pi(z)$$ 是上面的 permutation() )
            r"""
            $$
            \begin{eqnarray}
            \hspace{10em}z_0 &=& x_0\\
            \hspace{10em}z_1 &=& \big(x_1 - \mu_1(x_{<1})\big) \otimes \exp\big(-\alpha_1(x_{<1})\big)\\
            \hspace{10em}z_2 &=& \big(x_2 - \mu_2(x_{<2})\big) \otimes \exp\big(-\alpha_2(x_{<2})\big)\\
            \hspace{10em} &\cdots& \\
            \hspace{10em}z_{L-1} &=& \big(x_{L-1} - \mu_{L-1}(x_{<L-1})\big) \otimes \exp\big(-\alpha_{L-1}(x_{<L-1})\big)
            \end{eqnarray}
            $$
            """
            # 其中 $$ x_{<k} = [x_1,x_2,...,x_{k-1}] $$
            # 考虑到 attn_mask 是值全为1的下三角矩阵, transformer 第 k 个元素是 $$z_k$$ :
            # Attn[:,k-1,:] = $$z_{k-1}$$, $$k \in [1,L]$$
            # endregion
        x = self.proj_out(x)  # shape: (B, L, C_out)

        # TODO 弄清楚这一步是干什么的
        x = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)  # shape: (B, L, C_out)
        # region NOTE
        # zeor_like  -> (B, 1,   C_out)
        # x[:,:-1,:] -> (B, L-1, C_out)
        # concat     -> (B, L,   C_out)
        # 也就是令 $$z_0 = [0,...,0]^D$$
        # endregion

        # region 把输出拆分成两个部分: $$\alpha_i(x_{<i})$$ 和 $$\mu_i(x_{<i})$$
        if self.nvp:  # C_out = 2 * C_x
            x_alpha, x_mu = x.chunk(2, dim=-1)  # shape: (B, L, C_x), C_out = 2 * C_x
        else:  # C_out = C_x
            x_mu = x
            x_alpha = torch.zeros_like(x)  # shape: (B, L, C_x)
        # endregion

        scale = (-x_alpha.float()).exp().type(x_alpha.dtype)
        # region NOTE 计算缩放因子
        # 原论文公式(3) $$ \odot $$ 后的部分:
        # $$ \exp(-\alpha_i(x_{<i})) $$
        # endregion

        logdet = -x_alpha.mean(dim=[1, 2])  # shape: ()
        # region NOTE 求雅可比行列式的值
        #  原论文公式(5):
        #
        # $$ \log \big( |det(\frac{df(x)}{dx})| \big) = -\sum^{L-1}_{i = 0}\sum^{D-1}_{j =0} \alpha_i(x_{<i})_j $$
        #
        # endregion

        return self.permutation((x_hat - x_mu) * scale, inverse=True), logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i: i + 1]  # get i-th patch but keep the sequence dimension
        x = self.proj_in(x_in) + pos_embed[i: i + 1]
        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)
        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        self.set_sample_mode(True)
        T = x.size(1)
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')
            if guidance > 0 and guide_what:
                za_u, zb_u = self.reverse_step(x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond')
                if annealed_guidance:
                    g = (i + 1) / (T - 1) * guidance
                else:
                    g = guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)
