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
    """

    + 相较于 TarFlow 删除 NVP 相关设置, 默认启用 NVP
    """
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
        """设置 GSJ 模式.

        在设置时, 会改变内部所有 Attention 模块的 GSJmode 属性.

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
        self,x: torch.Tensor,y: torch.Tensor | None = None,
        hyper_parameters: ReverseHyperParameters = ReverseHyperParameters(),) -> torch.Tensor:
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
        B, L, C = x.size()  # shape: (B, L, C)

        # 置换操作, 在原论文内由 $$ \pi^{-1}(z) $$ 表示
        x = self.permutation(x)  # shape: (B, L, C)

        hyper_parameters = self._set_GSJmode(hyper_parameters, L)

        # 根据zeroguess(IGM)决定采用何种方式输入 x
        if hyper_parameters.zero_guess == 0:
            x_to_input = x.clone()  # shape: (B, L, C)
        else:
            x_to_input = torch.cat([x[:, :1, :], torch.zeros_like(x[:, 1:, :])], dim=1)

        match self.GSJmode:
            case "GS":  # TarFlow 的原始逆运算方式
                z = self._GS_reverse(x_to_input, y, hyper_parameters)
                z = self.permutation(z, inverse=True)  # shape: (B, L, C)
                pass
            case "J":  # GSJ 论文的 **纯并行** 逆运算方式
                z = self._J_reverse(x_to_input, y, x, hyper_parameters)
                z = self.permutation(z, inverse=True)  # shape: (B, L, C)
            case "GSJ":  # GSJ 论文的 **混合** 逆运算方式
                z = self._GSJ_reverse(x_to_input, y,x, hyper_paras=hyper_parameters)
                z = self.permutation(z, inverse=True)  # shape: (B, L, C)

        self.pos_embed = None  # type: ignore
        self.GSJmode = "J"
        return self.permutation(x, inverse=True)

    # region forward 的子步骤
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

        x_alpha, x_mu = x.chunk(2, dim=-1)  # shape: (B, L, C)

        # region NOTE NVP和非NVP
        # 在不启用NVP时, logdet 显然应该为 0, logdet 又和 $$\alpha_i(\cdot)$$ 有关.
        # 一个简单实现的方式是令 $$ \alpha_i(\cdot) =0 $$
        # 即上面 else 块的做法: 令 $$f_i(\cdot) = \mu_i(\cdot)$$, $$ \alpha_i(\cdot)$$
        # endregion
        return x_alpha, x_mu

    # endregion

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

    def _GS_reverse(self, x: torch.Tensor,
                    y: torch.Tensor | None = None,
                    hyper_paras: ReverseHyperParameters = ReverseHyperParameters()) -> torch.Tensor:
        """GS 逆运算
        逆运算的核心步骤, 逐行计算逆运算, 每行的逆运算依赖于前一行的结果.

        Args:
            x (torch.Tensor): 输入张量(序列), shape: (B, L, C)
            y (torch.Tensor | None, optional): 输入对应的标签, shape: (B,). Defaults to None.
            hyper_paras (ReverseHyperParameters, optional): 超参数. Defaults to ReverseHyperParameters().

        Returns:
            x (torch.Tensor): 逆运算后的张量(序列), shape: (B, L, C)
        """

        L = x.size(1)  # shape: (B, L, C), L为序列长度
        for i in range(L - 1):  # x 按行计算,每行 shape = (B, 1, C). 注意 共有 L-1 个元素, 这是由于 x[l-1] 不需要变更

            x_single_line = x[:, i:i + 1, :]  # 取出索引为 i 的行, shape: (B, 1, C)

            # 位置投影
            x_temp = self.proj_in(x_single_line) + self.pos_embed[i:i + 1, :]  # shape: (B,1,C_hidden)

            # 计算条件引导下的逆运算
            x_alpha_cond, x_mu_cond = self._reverse_substep(x_temp, y, jacobi_attn_mask=None, tau=hyper_paras.tau, which_cache='cond')

            x_alpha = x_alpha_cond
            x_mu = x_mu_cond

            # 计算非条件引导下的逆运算 , ab 指示引导 $$\alpha(x)$$ 和 $$\mu(x)$$ 的哪部分
            if hyper_paras.no_classification_guide:
                x_alpha_uncond, x_mu_uncond = self._reverse_substep(x_temp, None, jacobi_attn_mask=None, tau=hyper_paras.tau, which_cache='uncond')

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
            x[:, i + 1] = x[:, i + 1]* scale  + x_mu[:, 0]
        return x  # shape: (B, L, C)  # 返回逆运算后的结果

    def _J_reverse(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None,
            z: torch.Tensor,
            hyper_paras: ReverseHyperParameters = ReverseHyperParameters()) -> torch.Tensor:
        """Jacobi 逆运算

        Args:
            x (torch.Tensor): 输入张量(序列) 的 Jacobi chunk, shape: (B, J, C)
            y (torch.Tensor | None): 输入张量对应的标签, shape: (B,). Defaults to None.
            z (torch.Tensor): 本块原始输入张量(序列) 的 Jacobi chunk, shape: (B, J, C)
            hyper_paras (ReverseHyperParameters, optional): 超参数. Defaults to ReverseHyperParameters().

        Returns:
            x (torch.Tensor): shape: (B, J, C)
        """
        # 初始化
        B, J, C = x.shape
        x_current = x  # X迭代之前的初始值
        iter_count = 0  # 迭代次数
        diff = 1e6  # 差值, 用于判断迭代是否收敛

        # 当未满足迭代停止条件和收敛条件时, 使用 x_current 反复更新 x_next
        while iter_count < hyper_paras.max_jacobi and diff > hyper_paras.ebound:

            # region 用 x_current 计算 x_next, 等效于 x_next = func(x_current)
            x_next = self.proj_in(x_current) + self.pos_embed  # shape: (B, J, C_hidden)

            # 分类引导
            x_alpha_cond, x_mu_cond = self._reverse_substep(x_next, y, jacobi_attn_mask=self.attn_mask, tau=hyper_paras.tau, which_cache='cond')

            x_alpha = x_alpha_cond
            x_mu = x_mu_cond

            if hyper_paras.guidance > 0 and hyper_paras.guide_what:
                x_alpha_uncond, x_mu_uncond = self._reverse_substep(
                    x_next, None, jacobi_attn_mask=self.attn_mask, tau=hyper_paras.tau, which_cache='cond')

                # 确定引导权重 w_i
                if hyper_paras.annealed_guidance:
                    w_i: torch.Tensor = torch.arange(1, J + 1, device=z.device) / (J - 1) * hyper_paras.guidance
                    w_i = w_i.view(1, -1, 1)
                else:
                    w_i = hyper_paras.guidance  # type: ignore

                # 非条件引导
                if 'a' in hyper_paras.guide_what:
                    x_alpha = x_alpha_cond + w_i * (x_alpha_cond - x_alpha_uncond)
                if 'b' in hyper_paras.guide_what:
                    x_mu = x_mu_cond + w_i * (x_mu_cond - x_mu_uncond)

            # 把第 1 行设置为 0 (梯度截断)
            x_alpha = torch.cat([torch.zeros_like(x_alpha[:, :1, :]), x_alpha[:, :-1, :]], dim=1)
            x_mu = torch.cat([torch.zeros_like(x_mu[:, :1, :]), x_mu[:, :-1, :]], dim=1)

            x_next = ((x_alpha.float().exp().type(x_alpha.dtype)) * z + x_mu).clamp(-3, 3)
            # endregion

            # 更新 判断条件
            diff = torch.linalg.norm(x_next - x_current) / (B * C)
            iter_count += 1

            # 更新 x_current, 进入下一次迭代
            x_current = x_next
        return x_next

    def _GSJ_reverse(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None,
            z: torch.Tensor,
            hyper_paras: ReverseHyperParameters = ReverseHyperParameters(),
    ) -> torch.Tensor:
        base_attn_mask = torch.tril(
            torch.ones(
                hyper_paras.jacobi_size,
                hyper_paras.jacobi_size,
                device=z.device))  # 注意力 mask, 下三角全为1的矩阵, 用于屏蔽未来的信息, shape = (J, J)
        B, L, C = z.size()  # shape: (B, L, C)

        for i in range(hyper_paras.num_GS):  # 对于每个 jacobi 块, 对应原论文伪代码中的第 4 行

            # region NOTE
            # torch.cat 的第一部分: shape = (J, J) * i 个,
            # torch.cat 的第二部分: 下三角矩阵, shape = (J, J)
            # jacobi_attn_mask.shape: (J, (i+1)J )
            # jacobi_attn_mask 的作用:
            # 保留索引为 i 的 jacobi 块之前的块的所有信息,
            # 对索引为 i 的 jacobi 块内的按行保留信息
            # endregion
            jacobi_attn_mask = torch.cat([torch.ones(hyper_paras.jacobi_size, hyper_paras.jacobi_size, device=z.device)]
                                         * i + [base_attn_mask], dim=1)

            # 判断该行是不是最后一个 jacobi 块. 最后一个 jacobi 块长度可能不为 J
            is_last_jacobi_chunk: bool = (i == hyper_paras.num_GS - 1)

            # region 初始化 每个 Jacobi 块 进行 J 运算需要的 x, z, z对应的 pos_embed
            if is_last_jacobi_chunk:
                z_sub = z[:, -(hyper_paras.jacobi_size - 1):, :]  # 从原始 Z 中取出 最后一个 jacobi 块, shape: (B, J, C)
                pos_embed_sub = self.pos_embed[-hyper_paras.jacobi_size:-1]
                x_curr_sub = x[:, -(hyper_paras.jacobi_size - 1):].clone()
                jacobi_attn_mask = jacobi_attn_mask[:-1, :-1]  # shape: (J-1, (i+1)J-1 )
            else:
                z_sub = z[:, (i * hyper_paras.jacobi_size + 1):((i + 1) * hyper_paras.jacobi_size + 1),:]  # 从原始 Z 中取出 索引为 i 的 jacobi 块, shape: (B, J, C)
                pos_embed_sub = self.pos_embed[(i * hyper_paras.jacobi_size):((i + 1) * hyper_paras.jacobi_size)
                                               ]  # 从位置编码中取出 第 i 个 jacobi 块对应的位置编码, shape: (J, C_hidden)
                x_curr_sub = x[:, (i * hyper_paras.jacobi_size + 1):((i + 1) * hyper_paras.jacobi_size + 1)
                               ].clone()  # 从 x 中取出 索引为 i 的 jacobi 块作为 J运算的 x_current, shape: (B, J, C)
                # NOTE 注意, z 和 x 索引为0 的行不在任何 jacobi 块中
            # endregion

            if hyper_paras.incre1 and i == 0:  # 如果是第一个 jacobi 块, 且开启了 incre1 模式
                # BUG j 没有使用
                for j in range(hyper_paras.jacobi_size):  # 对于每一行
                    x_in = x[:, i: i + 1]  # BUG ? x_in 始终为 x[:,0,:], shape = (B,1,C)
                    x = self.proj_in(x_in) + self.pos_embed[i: i + 1]  # x变成了 (B,1,C_hidden)
                    x_alpha_cond, x_mu_cond = self._reverse_substep(x, y, jacobi_attn_mask=None, which_cache='cond')
                    if hyper_paras.guidance > 0 and hyper_paras.guide_what:
                        x_alpha_uncond, x_mu_uncond = self._reverse_substep(x, None, jacobi_attn_mask=None, tau=hyper_paras.tau, which_cache='uncond')
                        if hyper_paras.annealed_guidance:
                            w_i = (i + 1) / (L - 1) * hyper_paras.guidance
                        else:
                            w_i = hyper_paras.guidance
                        if 'a' in hyper_paras.guide_what:
                            x_alpha_cond = x_alpha_cond + w_i * (x_alpha_cond - x_alpha_uncond)
                        if 'b' in hyper_paras.guide_what:
                            x_mu_cond = x_mu_cond + w_i * (x_mu_cond - x_mu_uncond)
                    scale = x_alpha_cond[:, 0].float().exp().type(x_alpha_cond.dtype)  # get rid of the sequence dimension
                    x[:, i + 1] = z[:, i + 1] * scale + x_mu_cond[:, 0]
                    self._cat_kv_temp('cond')
                    if hyper_paras.guidance > 0 and hyper_paras.guide_what:
                        self._cat_kv_temp('uncond')
                continue

            # 设置迭代阈值
            n_iter = 0
            diff = 1e6

            while (n_iter < hyper_paras.max_jacobi) and (diff > hyper_paras.ebound):  # 当满足迭代条件时, 进行迭代计算
                # region 使用 x_current 计算 x_next

                # 由于最后一个 jacobi 块的形状不确定, 根据是否为最后一个jacobi块初始化 x_next
                # 为方便讨论, x_next 的shape 统一记为 (B, J, C)
                if is_last_jacobi_chunk:
                    x_next = x[:, -hyper_paras.jacobi_size:-1].clone()
                else:
                    x_next = x[:, (i * hyper_paras.jacobi_size):((i + 1) * hyper_paras.jacobi_size), :].clone()

                x_next = self.proj_in(x_next) + pos_embed_sub  # shape: (B, J, C_hidden)

                # 条件引导
                # BUG
                # x_next.shape: (B, J, C_hidden)
                # jacobi_attn_mask.shape: (J, (i+1)J)
                x_alpha_cond, x_mu_cond = self._reverse_substep(x_next, y, jacobi_attn_mask, which_cache='cond')

                # 非条件引导
                if hyper_paras.no_classification_guide:
                    x_alpha_uncond, x_mu_uncond = self._reverse_substep(x_next, None, jacobi_attn_mask, hyper_paras.tau, which_cache='uncond')

                    # 确定引导权重 w_i
                    if hyper_paras.annealed_guidance:
                        if is_last_jacobi_chunk:
                            w_i = torch.arange(L - hyper_paras.jacobi_size + 1, L, device=z.device) / (L - 1) * hyper_paras.guidance
                        else:
                            w_i = torch.arange(i * hyper_paras.jacobi_size + 1, (i + 1) *
                                               hyper_paras.jacobi_size + 1, device=z.device) / (L - 1) * hyper_paras.guidance
                        w_i = w_i.view(1, len(w_i), 1)
                    else:
                        w_i = hyper_paras.guidance
                    # 非条件引导
                    if 'a' in hyper_paras.guide_what:
                        x_alpha_cond = x_alpha_cond + w_i * (x_alpha_cond - x_alpha_uncond)
                    if 'b' in hyper_paras.guide_what:
                        x_mu_cond = x_mu_cond + w_i * (x_mu_cond - x_mu_uncond)
                scale = x_alpha_cond.float().exp().type(x_alpha_cond.dtype)
                x_next = (scale * z_sub + x_mu_cond).clamp(-3, 3)

                # 更新 z_next
                if is_last_jacobi_chunk:
                    x[:, -(hyper_paras.jacobi_size - 1):] = x_next
                else:
                    x[:, (i * hyper_paras.jacobi_size + 1):((i + 1) * hyper_paras.jacobi_size + 1)] = x_next
                # endregion
                diff = torch.norm(x_next - x_curr_sub) / (B * C)
                n_iter = n_iter + 1
                x_curr_sub = x_next
            self._cat_kv_temp('cond')
            if hyper_paras.guidance > 0 and hyper_paras.guide_what:
                self._cat_kv_temp('uncond')
        return x  # shape: (B, L, C)  # 返回逆运算后的结果

    def _cat_kv_temp(self, which_cache: str = 'cond'):
        for m in self.modules():
            if isinstance(m, Attention):
                m._k_cache[which_cache].append(m._k_gsj_cache[which_cache])  # type: ignore
                m._v_cache[which_cache].append(m._v_gsj_cache[which_cache])  # type: ignore

    def _set_GSJmode(self, hyper_paras: ReverseHyperParameters, L: int):
        num_GS = hyper_paras.num_GS
        # 若没有指定前多少行使用 GS 模式或指定全部行使用 GS 模式,
        # 则使用 TarFlow 原本的逆运算模式,
        # 即 GS 模式
        if num_GS < 1 or num_GS == L:
            self.GSJmode = "GS"
        # 若仅指定第一块使用 GS 模式, 意味着全部使用 J 模式
        # 即纯并行模式
        elif num_GS == 1:
            self.GSJmode = "J"
        # 若指定前某些块使用 GS 模式, 意味着部分使用 GS 模式, 部分使用 J 模式
        elif num_GS > 1:
            self.GSJmode = "GSJ"

        # 由于 num_GS <1 时行为和 num_GS= L 时相同, 令 num_GS = L 保持代码便于理解
        if num_GS < 1:
            hyper_paras.num_GS = L
            hyper_paras.max_jacobi = 1

        # 计算 $$J_L$$
        jacobi_size = L // num_GS
        hyper_paras.jacobi_size = jacobi_size
        return hyper_paras
