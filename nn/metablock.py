"""TarFlow 的 MetaBlock 模块. TarFlow的核心架构为多个 MetaBlock串联"""
from typing import TYPE_CHECKING, Literal

import torch

from config import MetaBlockConfig
from nn import ReverseHyperParameters
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
    # attn_mask
    # region TYPR_CHECKING
    if TYPE_CHECKING:
        def proj_in(self, x: torch.Tensor) -> torch.Tensor:
            r"""线性投影层(in).

            将输入的通道映射到计算注意力时的通道

            $$ Output_{(B,L,C-hidden)} = proj_{in}(Input_{(B,L,C)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (B, L, C_hidden)
            """
            ...

        def proj_out(self, x: torch.Tensor) -> torch.Tensor:
            r"""线性投影层(out).

            将计算注意力时的通道映射到输出的通道, 输出的通道数为 `C * (1 + nvp)`

            $$ Output_{(B,L,C * (1 + nvp))} = proj_{out}(Input_{(B,L,C-hidden)}) $$

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C_hidden)

            Returns:
                torch.Tensor: 维度映射后的输出张量, shape: (B, L, C * (1 + nvp) )
            """
            ...

        def __call__(self, x: torch.Tensor, logdet: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
            """前向传播

            Args:
                x (torch.Tensor): 输入张量, shape: (B, L, C)
                y (torch.Tensor | None, optional): 输入张量的标签, shape: (B). Defaults to None.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: 输出张量, shape: (B, L, C), 本层雅可比行列式的 log 值, shape: (B)
            """
            ...

        @property
        def attn_mask(self) -> torch.Tensor:
            """注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息

            shape: (L, L)
            """
            ...
    # endregion

    def __init__(self, config: MetaBlockConfig, permutation: Permutation):
        self._config = config
        super().__init__()
        self.proj_in = torch.nn.Linear(self._config.channels_in, self._config.channels_hidden)
        self.pos_embed_matrix = torch.nn.Parameter(torch.randn(self._config.num_patches, self._config.channels_hidden) * 1e-2)
        """位置嵌入编码使用的矩阵(可学习参数).

        `shape: (L, C_hidden)` """
        self.class_embed_matrix = torch.nn.Parameter(
            torch.randn(
                self._config.num_classes,
                1,
                self._config.channels_hidden) *
            1e-2) if self._config.num_classes else None
        """类别嵌入编码使用的矩阵(可学习参数).

        初始化时, 若`num_classes > 0`, 该参数用于存储每个类别的嵌入编码. 否则为 `None`.

        `shape: (num_classes, 1, C_hidden)`
        """
        self.attn_blocks: list[AttentionBlock] = torch.nn.ModuleList(
            [AttentionBlock(self._config.AttentionBlockConfig)
             for _ in range(self._config.num_layers)])  # type: ignore
        self.proj_out = torch.nn.Linear(self._config.channels_hidden, self._config.channels_in * (1 + self._config.nvp))
        self.proj_out.weight.data.fill_(0.0)
        self.permutation: Permutation = permutation
        """置换操作块."""
        self.register_buffer(
            'attn_mask',
            torch.tril(
                torch.ones(
                    self._config.num_patches,
                    self._config.num_patches)))  # 注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息
        # properties
        self._pos_embed: torch.Tensor = None  # type: ignore
        self._GSJmode: Literal["GS", "J", "GSJ"] = "J"

    # region properties
    @property
    def pos_embed(self) -> torch.Tensor:
        """shape: (L, C_hidden)

        Returns:
            permutatedpos_embed_matrix (torch.Tensor): 置换后的 pos_embed_matrix
        """
        if not self._pos_embed:
            self._pos_embed = self.permutation(self.pos_embed_matrix, dim=0)
        return self._pos_embed

    @pos_embed.setter
    def pos_embed(self, value: torch.Tensor):
        if value:
            self._pos_embed = value
        else:
            self._pos_embed = None  # type: ignore

    @property
    def GSJmode(self) -> Literal["GS", "J", "GSJ"]:
        return self._GSJmode

    @GSJmode.setter
    def GSJmode(self, value: Literal["GS", "J", "GSJ"]):
        """设置 GSJ 模式

        Args:
            value (Literal["GS", "J", "GSJ"]): GSJ 模式
        """
        if value not in ["GS", "J", "GSJ"]:
            raise ValueError("GSJmode must be one of 'GS', 'J', 'GSJ'")
        self._GSJmode = value
        for m in self.modules():
            if isinstance(m, Attention):
                m.GSJmode = value

    # endregion

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
        """前向传播

        Args:
            x (torch.Tensor): 输入张量, shape: (B, L, C)
            y (torch.Tensor | None, optional): 输入张量的标签, shape:(B). Defaults to None.

        Returns:
            output,logdet (tuple[torch.Tensor, torch.Tensor]): 输出张量, shape: (B, L, C), 本层雅可比行列式的 log 值, shape: (B)
        """
        # 置换操作, 在原论文内由 $$ \pi(z) $$ 表示
        x = self.permutation(x)  # shape: (B, L, C)

        # 缓存x备用
        x_hat = x  # shape: (B, L, C)

        # 位置嵌入编码 + 投影 in
        x = self.proj_in(x) + self.pos_embed  # shape: (B, L, C_hidden)

        # 分类引导处理
        x = self._classifier_guidance(x, y)

        # 计算注意力
        x = self._attention(x)

        # 投影 out
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
        x_alpha, x_mu = self._split_to_alpha_and_mu(x)

        scale = (-x_alpha.float()).exp().type(x_alpha.dtype)
        # region NOTE 计算缩放因子
        # 原论文公式(3) $$ \odot $$ 后的部分:
        # $$ \exp(-\alpha_i(x_{<i})) $$
        # endregion

        z = (x_hat - x_mu) * scale  # shape: (B, L, C)

        # add in GSJ paper
        d = {}
        if self._config.detect_mode:
            ign = self._calculateIGN(x_hat, z)
            crn = self._calculateCRN(x_alpha, x_hat)
            d["IGN"] = ign
            d["CRN"] = crn

        logdet = -x_alpha.mean(dim=[1, 2])  # shape: (B)
        # region NOTE 求雅可比行列式的值
        #  原论文公式(5):
        #
        # $$ \log \big( |det(\frac{df(x)}{dx})| \big) = -\sum^{L-1}_{i = 0}\sum^{D-1}_{j =0} \alpha_i(x_{<i})_j $$
        #
        # endregion

        self.pos_embed = None  # type: ignore

        return self.permutation(z, inverse=True), logdet, d

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        hyper_parameters: ReverseHyperParameters = ReverseHyperParameters(),
        show_trace: bool = False,
        X_target: torch.Tensor | None = None,
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
        B, L, C = x.size()  # shape: (B, L, C)
        x = self.permutation(x)  # shape: (B, L, C)

        hyper_parameters, jacobi_size = self._set_GSJmode(hyper_parameters, L)

        if hyper_parameters.zero_guess == 0:
            x_next = x.clone()  # shape: (B, L, C)
        else:
            x_next = torch.cat([x[:, :1, :], torch.zeros_like(x[:, 1:, :])], dim=1)

        match self.GSJmode:
            case "GS":  # TarFlow 的原始逆运算方式
                z = self._GS_reverse(x_next, y, hyper_parameters)
                z = self.permutation(z, inverse=True)  # shape: (B, L, C)
                pass
            case "J":  # GSJ 论文的 **纯并行** 逆运算方式
                z = self._J_reverse(x_next, y, x, hyper_parameters)
                z = self.permutation(z, inverse=True)  # shape: (B, L, C)
            case "GSJ":  # GSJ 论文的 **混合** 逆运算方式
                pass



        for i in range(L - 1):  # x按行计算,每行为 (B,1,C_hidden). 注意 共有L-1个元素, 这是由于 x_l-1 不需要变更

            # 计算条件引导下的逆运算
            z_alpha_cond, z_mu_cond = self._reverse_step(x, self.pos_embed, i, y, which_cache='cond')  # shape: (B,1,C_hidden)

            z_alpha = z_alpha_cond
            z_mu = z_mu_cond

            # 计算非条件引导下的逆运算
            if guidance > 0 and guide_what:
                z_alpha_uncond, z_mu_uncond = self._reverse_step(x, self.pos_embed, i, None, tau=tau, which_cache='uncond')

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

            # BUG exp 后可能会导致溢出, 使计算变为 NaN
            scale = z_alpha[:, 0].float().exp().type(z_alpha.dtype)  # shape: (B,C_hidden)

            # 上面计算的是第 i 行的逆运算, 替换原来的第 i 行
            x[:, i + 1] = x[:, i + 1] * scale + z_mu[:, 0]
        self.pos_embed = None  # type: ignore
        return self.permutation(x, inverse=True)

    def _reverse_step(
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
        # 注意, 使用 x_i[:,i]会使形状变为 (B,C), 这里需要保留维度
        x_i = x[:, i: i + 1]  # shape: (B,1,C).

        # 位置投影
        x_i = self.proj_in(x_i) + pos_embed[i: i + 1]  # shape: (B, 1, C_hidden)

        # 类型引导
        if self.class_embed_matrix is not None:  # 有分类引导
            if y is not None:  # x_i 有标签
                x_i = x_i + self.class_embed_matrix[y]  # shape: (B, 1, C_hidden)
            else:  # x_i 没有标签
                x_i = x_i + self.class_embed_matrix.mean(dim=0)
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


    def _calculateIGN(self, x_star: torch.Tensor, Z: torch.Tensor) -> list[torch.Tensor]:
        """计算 IGN 指数

        Args:
            x_star (torch.Tensor): X*
            Z (torch.Tensor): X* `forward` 计算得到的 Z


        Returns:
            IGN指数 (list[torch.Tensor]): Z 和 Z0的 IGN指数
        """
        z0 = torch.cat([x_star[:, :1, :], torch.zeros_like(x_star[:, 1:, :])], dim=1)
        IGN = []
        for z_ in [Z, z0]:

            # region 把 $$Z$$ 和 $$Z_0$$ 分别作为 $$ X^{(0)}$$, 代入 $$\sum(X^{(0)})Z + \mu(X^{(0)}) - X^*$$
            z_ = self.proj_in(z_) + self.pos_embed
            if self.class_embed_matrix is not None:
                z_ = z_ + self.class_embed_matrix.mean(dim=0)
            for block in self.attn_blocks:
                z_ = block(z_, self.attn_mask)
            z_ = self.proj_out(z_)

            z_ = torch.cat([torch.zeros_like(z_[:, :1, :]), z_[:, :-1, :]], dim=1)

            z_alpha, z_mu = z_.chunk(2, dim=-1)

            scale = (z_alpha.float()).exp().type(z_alpha.dtype)
            res = (scale * Z + z_mu) - x_star
            # endregion

            # 在 Batch 上求平均得到平均 (L,C), 再求矩阵 (L,C) 的 norm
            singular_value: torch.Tensor = torch.linalg.norm(res.mean(dim=0), ord=self._config.norm)
            IGN.append(singular_value.item())
        return IGN

    def _calculateCRN(self, alpha: torch.Tensor, x_star: torch.Tensor) -> list[torch.Tensor]:
        CRN = []

        # 计算 $$||{\sum}^{-1}(X)X||_2$$
        CRN.append(torch.linalg.norm((alpha * x_star).sum(dim=0), ord=self._config.norm).item())

        # 计算 $$||W_s||_2$$ 和 $$||W_u||_2$$
        W: torch.Tensor = self.proj_out.weight
        W_s, W_u = W.chunk(2, dim=0)
        CRN.append(torch.linalg.norm(W_s, ord=self._config.norm).item())
        CRN.append(torch.linalg.norm(W_u, ord=self._config.norm).item())

        # 计算最终的CRM $$||{\sum}^{-1}(X)X||_2 * ||W_s||_2 + ||W_u||_2$$
        CRN.append(CRN[0] * CRN[1] + CRN[2])
        return CRN

    def _classifier_guidance(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.class_embed_matrix is not None:  # 有分类引导

            if y is not None:  # 有分类标签

                if (y < 0).any():  # 存在负标签
                    m = (y < 0).float().view(-1, 1, 1)  # 遮罩, 标签 <0 为1, 否则为0. shape: (B, 1, 1)
                    class_embed = (1 - m) * self.class_embed_matrix[y] + m * self.class_embed_matrix.mean(dim=0)
                    # (1-m) :shape (B, 1, 1)
                    #
                else:
                    class_embed = self.class_embed_matrix[y]  # shape: (B, 1, C_hidden)
                x = x + class_embed

            else:  # 没有分类引导

                x = x + self.class_embed_matrix.mean(dim=0)
        else:
            pass  # 没有分类引导, 不做任何操作
        return x

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
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
        return x

    def _split_to_alpha_and_mu(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._config.nvp:
            x_alpha, x_mu = x.chunk(2, dim=-1)  # shape: (B, L, C)
        else:
            x_mu = x
            x_alpha = torch.zeros_like(x)  # shape: (B, L, C)
        # region NOTE NVP和非NVP
        # 在不启用NVP时, logdet 显然应该为 0, logdet 又和 $$\alpha_i(\cdot)$$ 有关.
        # 一个简单实现的方式是令 $$ \alpha_i(\cdot) =0 $$
        # 即上面 else 块的做法: 令 $$f_i(\cdot) = \mu_i(\cdot)$$, $$ \alpha_i(\cdot)$$
        # endregion
        return x_alpha, x_mu

    def _reverse_substep(self,
                         x: torch.Tensor,
                         y: torch.Tensor | None = None,
                         jacobi_attn_mask: torch.Tensor | None = None,
                         tau: float = 1.0,
                         which_cache: Literal["cond", "uncond"] = "cond",

                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """逆运算的子步

        Args:
            x (torch.Tensor): 输入X, shape = (B,L,C)
            y (torch.Tensor | None, optional): 输入的标签, shape=(B,L,C). Defaults to None.
            jacobi_attn_mask (torch.Tensor | None, optional): 注意力mask. Defaults to None.
            attn_temp (float, optional): 注意力缩放因子. Defaults to 1.0.
            which_cache (Literal[&quot;cond&quot;, &quot;uncond&quot;], optional): 使用何种缓存. Defaults to "cond".

        Returns:
            逆运算的两个部分 (tuple[torch.Tensor, torch.Tensor]): x_alpha 和 x_mux_mu
        """

        # 分类引导
        if self.class_embed_matrix is not None:
            if y is not None:
                x = x + self.class_embed_matrix[y]
            else:
                x = x + self.class_embed_matrix.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, jacobi_attn_mask, tau, which_cache)

        x = self.proj_out(x)
        x_alpha, x_mu = x.chunk(2, dim=-1)
        return x_alpha, x_mu
        pass

    def _GS_reverse(self, x: torch.Tensor,
                    y: torch.Tensor | None = None,
                    hyper_paras: ReverseHyperParameters = ReverseHyperParameters()) -> torch.Tensor:
        L = x.size(1)  # shape: (B, L, C), L为序列长度
        for i in range(L - 1):  # x 按行计算,每行 shape = (B, 1, C). 注意 共有 L-1 个元素, 这是由于 x[l-1] 不需要变更

            x_in = x[:, i:i + 1, :]  # 取出索引为 i 的行, shape: (B, 1, C)

            # 位置投影
            x = self.proj_in(x_in) + self.pos_embed[i:i + 1, :]  # shape: (B,1,C_hidden)

            # 计算条件引导下的逆运算
            x_alpha_cond, x_mu_cond = self._reverse_substep(x, y, jacobi_attn_mask=None, tau=1.0, which_cache='cond')

            x_alpha = x_alpha_cond
            x_mu = x_mu_cond

            # 计算非条件引导下的逆运算 , ab 指示引导 $$\alpha(x)$$ 和 $$\mu(x)$$ 的哪部分
            if hyper_paras.no_classification_guide:
                x_alpha_uncond, x_mu_uncond = self._reverse_substep(x, None, jacobi_attn_mask=None, tau=hyper_paras.tau, which_cache='uncond')

                # 确定引导权重 w_i
                if hyper_paras.annealed_guidance:
                    # region NOTE
                    # 在原论文公式(11)后, 为:
                    #
                    # $$ w_i = \frac{i+1}{L-1}w $$
                    #
                    # endregion
                    w_i = (i + 1) / (L - 1) * hyper_paras.guidance
                else:
                    w_i = hyper_paras.guidance

                # 非条件引导
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
                if 'a' in hyper_paras.guide_what:
                    x_alpha = x_alpha_cond + w_i * (x_alpha_cond - x_alpha_uncond)
                if 'b' in hyper_paras.guide_what:
                    x_mu = x_mu_cond + w_i * (x_mu_cond - x_mu_uncond)

            scale = x_alpha[:, 0].float().exp().type(x_alpha.dtype)  # shape: (B, C_hidden)

            # 上面计算的是第 i 行的逆运算, 替换原来的第 i 行
            x[:, i + 1] = x[:, i + 1] * scale + x_mu[:, 0]
        return x  # shape: (B, L, C)  # 返回逆运算后的结果

    def _J_reverse(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None,
            x_origin: torch.Tensor,
            hyper_paras: ReverseHyperParameters = ReverseHyperParameters()) -> torch.Tensor:
        # 初始化
        B, L, C = x.shape
        x_current = x_next = x
        iter_count = 0  # 迭代次数
        diff = 1e6  # 差值, 用于判断迭代是否收敛
        iter_trace = []  # 迭代轨迹, 用于调试

        while iter_count < hyper_paras.max_jacobi and diff > hyper_paras.ebound:  # 当未满足迭代停止条件和收敛条件时, 使用x_current反复更新 x_next
            x_next = self.proj_in(x_current) + self.pos_embed  # shape: (B, L, C_hidden)

            # 分类引导
            x_alpha_cond, x_mu_cond = self._reverse_substep(x_next, y, jacobi_attn_mask=self.attn_mask, tau=hyper_paras.tau, which_cache='cond')

            x_alpha = x_alpha_cond
            x_mu = x_mu_cond

            if hyper_paras.guidance > 0 and hyper_paras.guide_what:
                # [TODO] 检查是否用 "cond"
                x_alpha_uncond, x_mu_uncond = self._reverse_substep(
                    x_next, None, jacobi_attn_mask=self.attn_mask, tau=hyper_paras.tau, which_cache='cond')

                # 确定引导权重 w_i
                if hyper_paras.annealed_guidance:
                    w_i: torch.Tensor = torch.arange(1, L + 1, device=x_origin.device) / (L - 1) * hyper_paras.guidance
                    w_i = w_i.view(1, -1, 1)
                else:
                    w_i = hyper_paras.guidance

                # 非条件引导
                if 'a' in hyper_paras.guide_what:
                    x_alpha = x_alpha_cond + w_i * (x_alpha_cond - x_alpha_uncond)
                if 'b' in hyper_paras.guide_what:
                    x_mu = x_mu_cond + w_i * (x_mu_cond - x_mu_uncond)

            # 把除第1行外的结果设置为 0
            x_alpha = torch.cat([torch.zeros_like(x_alpha[:, :1, :]), x_alpha[:, :-1, :]], dim=1)
            x_mu = torch.cat([torch.zeros_like(x_mu[:, :1, :]), x_mu[:, :-1, :]], dim=1)

            x_next = ((x_alpha.float().exp().type(x_alpha.dtype)) * x_origin + x_mu).clamp(-3, 3)

            # 更新 判断条件
            diff = torch.linalg.norm(x_next - x_current) / (B * C)
            iter_count += 1

            # 更新 x_current, 进入下一次迭代
            x_current = x_next
        return x_next

    def _GSJ_reverse(
            self,
            x: torch.Tensor, x_next: torch.Tensor, z: torch.Tensor,
            y: torch.Tensor | None = None,
            incre1: bool = False,
            max_jacobi: int = 100,
            ebound: float = 1e-8,
            num_GS: int = 1,
            guidance: float = 0,
            jacobi_size: int = 1,
            guide_what: str = 'ab',
            attn_temp: float = 1.0,
            annealed_guidance: bool = False) -> torch.Tensor:
        base_attn_mask = torch.tril(
            torch.ones(
                jacobi_size,
                jacobi_size, device=z.device))  # 注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息
        B, L, C = x.size()  # shape: (B, L, C)
        for i in range(num_GS):
            jacobi_attn_mask = torch.cat([torch.ones(jacobi_size, jacobi_size, device=z.device)] * i + [base_attn_mask], dim=1)
            last = (i == num_GS - 1)
            if last:
                z_sub = z[:, -(jacobi_size - 1):, :]
                pos_embed_sub = self.pos_embed[-jacobi_size:-1]
                x_curr_sub = x_next[:, -(jacobi_size - 1):].clone()
                jacobi_attn_mask = jacobi_attn_mask[:-1, :-1]
            else:
                z_sub = z[:, (i * jacobi_size + 1):((i + 1) * jacobi_size + 1)]
                pos_embed_sub = self.pos_embed[(i * jacobi_size):((i + 1) * jacobi_size)]
                x_curr_sub = x_next[:, (i * jacobi_size + 1):((i + 1) * jacobi_size + 1)].clone()

            if incre1 and i == 0:
                for j in range(jacobi_size):
                    x_in = x_next[:, i: i + 1]
                    x = self.proj_in(x_in) + self.pos_embed[i: i + 1]
                    xa, xc = self._reverse_substep(x, y, jacobi_attn_mask=None, which_cache='cond')
                    if guidance > 0 and guide_what:
                        xa_u, xc_u = self._reverse_substep(x, None, jacobi_attn_mask=None, tau=attn_temp, which_cache='uncond')
                        if annealed_guidance:
                            g = (i + 1) / (L - 1) * guidance
                        else:
                            g = guidance
                        if 'a' in guide_what:
                            xa = xa + g * (xa - xa_u)
                        if 'b' in guide_what:
                            xc = xc + g * (xc - xc_u)
                    alpha = xa[:, 0].float().exp().type(xa.dtype)  # get rid of the sequence dimension
                    x_next[:, i + 1] = z[:, i + 1] * alpha + xc[:, 0]
                    self.cat_kv_temp('cond')
                    if guidance > 0 and guide_what:
                        self.cat_kv_temp('uncond')
                continue

            n_iter = 0
            diff = 1e6
            iter_trace_module = []

            while (n_iter < max_jacobi) and (diff > ebound):
                if last:
                    x_next_sub = x_next[:, -jacobi_size:-1].clone()
                else:
                    x_next_sub = x_next[:, (i * jacobi_size):((i + 1) * jacobi_size)].clone()
                x_next_sub = self.proj_in(x_next_sub) + pos_embed_sub

                xa, xc = self._reverse_substep(x_next_sub, y, jacobi_attn_mask, which_cache='cond')
                if guidance > 0 and guide_what:
                    xa_u, xc_u = self._reverse_substep(x_next_sub, None, jacobi_attn_mask, attn_temp, which_cache='uncond')
                    if annealed_guidance:
                        if last:
                            g = torch.arange(L - jacobi_size + 1, L, device=z.device) / (L - 1) * guidance
                        else:
                            g = torch.arange(i * jacobi_size + 1, (i + 1) * jacobi_size + 1, device=z.device) / (L - 1) * guidance
                        g = g.view(1, len(g), 1)
                    else:
                        g = guidance
                    if 'a' in guide_what:
                        xa = xa + g * (xa - xa_u)
                    if 'c' in guide_what:
                        xc = xc + g * (xc - xc_u)
                alpha = xa.float().exp().type(xa.dtype)
                gamma = xc
                x_next_sub = alpha * z_sub + gamma
                x_next_sub = torch.clamp(x_next_sub, min=-3, max=3)

                if last:
                    x_next[:, -(jacobi_size - 1):] = x_next_sub
                    # if show_trace:
                    #     iter_trace_module.append((torch.norm(X_target[:, -(jacobi_size - 1):] - \
                    #                                 x_next_sub).item()) / (B * x_next_sub.size(1) * C))
                    #     print(n_iter)
                else:
                    x_next[:, (i * jacobi_size + 1):((i + 1) * jacobi_size + 1)] = x_next_sub
                    # if show_trace:
                    #     iter_trace_module.append(
                    #         (torch.norm(X_target[:, (i * jacobi_size + 1):((i + 1) * jacobi_size + 1)] - x_next_sub).item()) / (B * x_next_sub.size(1) * C))
                    #     print(n_iter)

                diff = torch.norm(x_next_sub - x_curr_sub) / (B * C)
                n_iter = n_iter + 1
                x_curr_sub = x_next_sub
            # if show_trace:
            #     iter_trace.append(iter_trace_module)
            self.cat_kv_temp('cond')
            if guidance > 0 and guide_what:
                self.cat_kv_temp('uncond')
        return x_next  # shape: (B, L, C)  # 返回逆运算后的结果
        pass

    def cat_kv_temp(self, which_cache: str = 'cond'):
        for m in self.modules():
            if isinstance(m, Attention):
                m._k_cache[which_cache].append(m._k_cache[which_cache + '_temp'])  # type: ignore
                m._v_cache[which_cache].append(m._v_cache[which_cache + '_temp'])  # type: ignore

    def _set_GSJmode(self, hyper_paras: ReverseHyperParameters, L: int):
        num_GS = hyper_paras.num_GS
        if num_GS < 1 or num_GS == L:
            self.GSJmode = "GS"
        elif num_GS == 1:
            self.GSJmode = "J"
        elif num_GS > 1:
            self.GSJmode = "GSJ"

        if num_GS < 1:
            hyper_paras.num_GS = L
            hyper_paras.max_jacobi = 1

        jacobi_size = L // num_GS
        return hyper_paras, jacobi_size
