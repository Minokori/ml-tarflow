"""TarFlow 的 MetaBlock 模块. TarFlow的核心架构为多个 MetaBlock串联"""
from typing import TYPE_CHECKING, Literal, overload

import torch

from nn.basemodule import *
from nn.blockmodule import *


# 符号说明:
# B: 批量大小 batch size
# L: 序列长度 (sequence length)
# C: 通道数 (channel size)
# H: 注意力头数 (number of attention heads)
# D: 每个注意力头的维度 (dimension of each attention head)
# H * D =C
# C_hidden: 隐藏层通道数

class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor
    """注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息

    shape: (L, L)
    """
    # region TYPR_CHECKING
    if TYPE_CHECKING:

        @overload
        def proj_in(self, x: torch.Tensor) -> torch.Tensor:
            """线性投影层(in).

            将输入的通道映射到计算注意力时的通道

            $$ Output_{(B,L,C-hidden)} = proj_{in}(Input_{(B,L,C)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (B, L, C_hidden)
            """
            ...

        @overload
        def proj_out(self, x: torch.Tensor) -> torch.Tensor:
            """线性投影层(out).

            将计算注意力时的通道映射到输出的通道, 输出的通道数为 `C * (1 + nvp)`

            $$ Output_{(B,L,C * (1 + nvp))} = proj_{out}(Input_{(B,L,C-hidden)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C_hidden)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (B, L, C * (1 + nvp) )
            """
            ...

        @overload
        def __call__(self, x: torch.Tensor, logdet: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)
                y (torch.Tensor | None, optional): 输入张量的标签, shape: (B). Defaults to None.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: 输出张量, shape: (B, L, C), 本层雅可比行列式的 log 值, shape: (B)
            """
            ...
    # endregion

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
            in_channels (int): 输入张量的通道数, 记作 `C`
            channels (int): 隐藏层通道数, 记作 `C_hidden`
            num_patches (int): 图像被分割成的块数, 也即序列长度 `L`
            permutation (Permutation): 置换类型.
            num_layers (int, optional): 注意力块数. Defaults to 1.
            head_dim (int, optional): 每个注意力头分配的通道数. Defaults to 64.
            expansion (int, optional): MLP的隐藏层扩大系数. Defaults to 4.
            nvp (bool, optional): 是否使用 `NVP` 模式. Defaults to True.
            num_classes (int, optional): 样本类别数, 设置为 0 即为没有类别 Defaults to 0.
        """
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        """位置嵌入编码使用的矩阵(可学习参数).

        `shape: (L, C_hidden)` """

        self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2) if num_classes else None
        """类别嵌入编码使用的矩阵(可学习参数).

        初始化时, 若`num_classes > 0`, 该参数用于存储每个类别的嵌入编码. 否则为 `None`.

        `shape: (num_classes, 1, C_hidden)`
        """
        self.attn_blocks: list[AttentionBlock] = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion)
             for _ in range(num_layers)])
        self.nvp = nvp
        """是否使用 `NVP (non-volume preserving)`, 非体积保持模式.

        非体积保持启用时, logdet 不为1
        """

        self.proj_out = torch.nn.Linear(channels, in_channels * (1 + nvp))
        self.proj_out.weight.data.fill_(0.0)

        self.permutation: Permutation = permutation
        """置换操作块."""

        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))  # 注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量, shape: (B, L, C)
            y (torch.Tensor | None, optional): 输入张量的标签, shape:(B). Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 输出张量, shape: (B, L, C), 本层雅可比行列式的 log 值, shape: (B)
        """
        # 置换操作, 在原论文内由 $$ \pi(z) $$ 表示
        x = self.permutation(x)  # shape: (B, L, C)

        # 缓存x备用
        x_hat = x  # shape: (B, L, C)

        # 位置嵌入编码
        pos_embed = self.permutation(self.pos_embed, dim=0)  # shape: (L, C_hidden)
        x = self.proj_in(x) + pos_embed  # shape: (B,L,C_hidden)

        # classifier guidance 和 classifier free guidance
        if self.class_embed is not None:  # 有分类引导

            if y is not None:  # 有分类标签

                if (y < 0).any():  # 存在负标签
                    m = (y < 0).float().view(-1, 1, 1)  # 遮罩, 标签 <0 为1, 否则为0. shape: (B, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                    # (1-m) :shape (B, 1, 1)
                    #
                else:
                    class_embed = self.class_embed[y]  # shape: (B, 1, C_hidden)
                x = x + class_embed

            else:  # 没有分类引导

                x = x + self.class_embed.mean(dim=0)
        else:
            pass  # 没有分类引导, 不做任何操作

        # 计算注意力
        for block in self.attn_blocks:
            x = block(x, self.attn_mask)  # shape: (B, L, C_hidden)
            # region NOTE ⚠ 结合 attn_mask, attention 的作用 ⚠
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
            # 考虑到 attn_mask 是值全为1的下三角矩阵, transformer 第 k 个元素仅由 第0~第k-1 个元素决定
            # 因此, 可以认为 Attention 的每一行在做如下操作:
            # $$line_k = f_k(x_{<k})$$
            # 对应上式公式(3), 可以把 Attention 的操作 $$f_k(x_{<k})$$ 以某种形式拆分成两部分, 一部分当作 $$\mu_k(x_{<k}) $$, 一部分当作 $$ \alpha_k(x_{<k}) $$
            # endregion
        x = self.proj_out(x)  # shape: (B, L, C * (1+nvp) )

        # 梯度断裂
        x = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)  # shape: (B, L, C * (1+nvp))
        # region NOTE
        # zeor_like  -> (B, 1,   C * (1+nvp) )
        # x[:,:-1,:] -> (B, L-1, C * (1+nvp) )
        # concat     -> (B, L,   C * (1+nvp) )
        # 也就是令 $$f_0 = [0,...,0]^D$$
        # 参照 https://github.com/apple/ml-tarflow/issues/8 , 原作者的回复如下:
        # "The first position $$z_0$$ goes through an identity transformation
        #  and zero padding is an easy way of doing it."
        #  由于 $$z_0$$ 事实上和 attention 中的参数无关, 也因此没有梯度联系, 因此把他手动置为0,以表示这种梯度的断裂.
        # endregion

        # 把输出 $$f_i(x_{<i})$$ 拆分成两个部分: $$\alpha_i(x_{<i})$$ 和 $$\mu_i(x_{<i})$$
        if self.nvp:
            x_alpha, x_mu = x.chunk(2, dim=-1)  # shape: (B, L, C)
        else:
            x_mu = x
            x_alpha = torch.zeros_like(x)  # shape: (B, L, C)
        # region NOTE NVP和非NVP
        # 在不启用NVP时, logdet 显然应该为 0, logdet 又和 $$\alpha_i(\cdot)$$ 有关.
        # 一个简单实现的方式是令 $$ \alpha_i(\cdot) =0 $$
        # 即上面 else 块的做法: 令 $$f_i(\cdot) = \mu_i(\cdot)$$, $$ \alpha_i(\cdot)$$
        # endregion

        scale = (-x_alpha.float()).exp().type(x_alpha.dtype)
        # region NOTE 计算缩放因子
        # 原论文公式(3) $$ \odot $$ 后的部分:
        # $$ \exp(-\alpha_i(x_{<i})) $$
        # endregion

        logdet = -x_alpha.mean(dim=[1, 2])  # shape: (B)
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
        tau: float = 1.0,
        which_cache: Literal["cond", "uncond"] = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单行 X 的反向传播

        Args:
            x (torch.Tensor): 输入张量(序列), shape: (B, L, C)
            pos_embed (torch.Tensor): 位置编码矩阵, shape: (L, C_hidden)
            i (int): 当前的行数 (第i行只能看到第i行之前的信息)
            y (torch.Tensor | None, optional): 输入张量对应的标签, shape: (B). Defaults to None.
            tau (float, optional): 手动注入温度项. Defaults to 1.0.
            which_cache (str, optional): 使用的缓存. Defaults to 'cond'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: x_alpha, x_mu
        """

        # 获得序列的第 i 个元素, i<=0<L
        x_i = x[:, i: i + 1]  # shape: (B,1,C).
        # 注意, 使用 x_i[:,i]会使形状变为 (B,C), 这里需要保留维度
        x_i = self.proj_in(x_i) + pos_embed[i: i + 1]  # shape: (B, 1, C_hidden)

        # 类型引导
        if self.class_embed is not None:  # 有分类引导
            if y is not None:  # x_i 有标签
                x_i = x_i + self.class_embed[y]  # shape: (B, 1, C_hidden)
            else:  # x_i 没有标签
                x_i = x_i + self.class_embed.mean(dim=0)
        else:  # 没有分类引导, 不做任何操作
            pass

        # 计算注意力, 相当于得到 $$ f(x) $$
        for block in self.attn_blocks:
            x_i = block(x_i, tau=tau, which_cache=which_cache)  # here we use kv caching, so no attn_mask
            # region NOTE key, value 缓存的作用
            # 由于每次只计算一行 x_i, k,v中保存了前i行的 k,v,和 mask效果相同
            # endregion
        x_i = self.proj_out(x_i)  # shape: (B, 1, C * (1+nvp) )

        # 将输出 $$ f(x_{<i}) $$ 拆成两部分: $$\alpha_i(x_{<i})$$ 和 $$\mu_i(x_{<i})$$
        if self.nvp:
            x_alpha, x_mu = x_i.chunk(2, dim=-1)  # shape: (B, 1, C)
        else:
            x_mu = x_i
            x_alpha = torch.zeros_like(x_i)  # shape: (B, 1, C)
        # region NOTE NVP和非NVP
        # 在不启用NVP时, logdet 显然应该为 0, logdet 又和 $$\alpha_i(\cdot)$$ 有关.
        # 一个简单实现的方式是令 $$ \alpha_i(\cdot) =0 $$
        # 即上面 else 块的做法: 令 $$f_i(\cdot) = \mu_i(\cdot)$$, $$ \alpha_i(\cdot) $$
        # endregion
        return x_alpha, x_mu

    def set_sample_mode(self, flag: bool = True):
        """设置是否为采样(逆运算)模式

        清空 Attention 块内的 k, v矩阵的缓存.

        Args:
            flag (bool, optional): 是否为采样模式. Defaults to True.
        """
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
        tau: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): 输入张量(序列), shape: (B, L, C)
            y (torch.Tensor | None, optional): 输入张量对应的标签, shape:(B) Defaults to None.
            guidance (float, optional): 引导权重 w. Defaults to 0.
            guide_what (str, optional): _description_. Defaults to 'ab'.
            tau (float, optional): 手动注入温度项. Defaults to 1.0.
            annealed_guidance (bool, optional): 是否使用退火引导权重 (使固定的 w 变成动态的 w(i,L) ). Defaults to False.

        Returns:
            torch.Tensor: 输出张量(序列), shape: (B,L,C)
        """

        # 置换操作, 在原论文内由 $$ \pi^{-1}(z) $$ 表示
        x = self.permutation(x)  # shape: (B, L, C)
        pos_embed = self.permutation(self.pos_embed, dim=0)  # shape: (L, C_hidden)
        self.set_sample_mode(True)
        L = x.size(1)
        for i in range(L - 1):  # x按行计算,每行为 (B,1,C_hidden). 注意 共有L-1个元素, 这是由于 x_l-1 不需要变更

            # 计算条件引导下的逆运算
            z_alpha_cond, z_mu_cond = self.reverse_step(x, pos_embed, i, y, which_cache='cond')  # shape: (B,1,C_hidden)

            z_alpha = z_alpha_cond
            z_mu = z_mu_cond

            # 计算非条件引导下的逆运算
            if guidance > 0 and guide_what:
                z_alpha_uncond, z_mu_uncond = self.reverse_step(x, pos_embed, i, None, tau=tau, which_cache='uncond')

                # 确定引导权重 w_i
                if annealed_guidance:
                    w_i = (i + 1) / (L - 1) * guidance
                    # region NOTE
                    # 在原论文公式(11)后, 为:
                    #
                    # $$ w_i = \frac{i+1}{L-1}w $$
                    #
                    # endregion
                else:
                    w_i = guidance

                # 非条件引导
                if 'a' in guide_what:
                    z_alpha = z_alpha_cond + w_i * (z_alpha_cond - z_alpha_uncond)
                if 'b' in guide_what:
                    z_mu = z_mu_cond + w_i * (z_mu_cond - z_mu_uncond)
                # shape: (B,1,C_hidden)
                # region NOTE
                # 对应原论文公式(11):
                r"""
                $$
                \begin{eqnarray}
                \hspace{8em}\alpha_i(z_{<i};\tau,w) &=& (1+w) \alpha_i(z_{<i};1) - w\alpha_i(z_{<i},\tau)\\
                \hspace{8em}\mu_i(z_{<i};\tau,w) &=& (1+w) \mu_i(z_{<i};1) - w\mu_i(z_{<i},\tau)
                \end{eqnarray}
                $$
                """
                # endregion

            scale = z_alpha[:, 0].float().exp().type(z_alpha.dtype)  # shape: (B,C_hidden)

            # 上面计算的是第 i 行的逆运算, 替换原来的第 i 行
            x[:, i + 1] = x[:, i + 1] * scale + z_mu[:, 0]
        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)
