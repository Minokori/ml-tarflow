from dataclasses import dataclass
from functools import cached_property

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class MLPConfig():
    channels: int
    """输入输出的维度"""
    expansion: int
    """扩大系数. 隐藏层维度 = 输入维度 * 扩大系数"""

    @cached_property
    def channels_hidden(self) -> int:
        """隐藏层的维度"""
        return self.channels * self.expansion


@dataclass_json
@dataclass
class AttentionConfig():
    channels_in: int
    """输入的维度 `C`"""
    channels_per_head: int
    """每个注意力头分配的维度 `D`"""

    @cached_property
    def num_heads(self) -> int:
        """计算注意力头的数量 `H`"""
        return self.channels_in // self.channels_per_head

    def post_init(self):
        """检查输入维度是否可以被每个注意力头分配的维度整除"""
        assert self.channels_in % self.channels_per_head == 0, f"输入维度 `{self.channels_in}` 无法被为每个注意力头分配的维度 `{self.channels_per_head}` 整除"


@dataclass_json
@dataclass
class AttentionBlockConfig():
    channels_in: int
    """输入张量的通道数 `C`"""
    channels_per_head: int
    """每个注意力头分配的通道数 `D`"""
    expansion: int = 4
    """MLP的隐藏层扩大系数."""

    def post_init(self):
        """检查输入维度是否可以被每个注意力头分配的维度整除"""
        assert self.channels_in % self.channels_per_head == 0, f"输入维度 `{self.channels_in}` 无法被为每个注意力头分配的维度 `{self.channels_per_head}` 整除"

    @cached_property
    def AttentionConfig(self) -> AttentionConfig:
        return AttentionConfig(channels_in=self.channels_in, channels_per_head=self.channels_per_head)

    @cached_property
    def MLPConfig(self) -> MLPConfig:
        return MLPConfig(channels=self.channels_in, expansion=self.expansion)


@dataclass_json
@dataclass
class MetaBlockConfig():
    channels_in: int
    """输入张量的通道数, 记作 `C`"""
    channels_hidden: int
    """隐藏层通道数, 记作 `C_hidden`"""
    num_patches: int
    """图像被分割成的块数, 也即序列长度 `L`"""
    num_layers: int = 1
    """注意力块数."""
    channels_per_head: int = 64
    """每个注意力头分配的通道数"""
    expansion: int = 4
    """MLP的隐藏层扩大系数."""
    nvp: bool = True
    """是否使用 `NVP` 模式"""
    num_classes: int = 0
    """样本类别数, 设置为 0 即为没有类别"""
    # Added in GSJ paper
    detect_mode: bool = False
    """`GSJ论文中增加的参数`, 是否为检测模式"""
    norm: int = 2
    """`GSJ论文中增加的参数`, 归一化范数"""

    @cached_property
    def AttentionBlockConfig(self) -> AttentionBlockConfig:
        return AttentionBlockConfig(channels_in=self.channels_hidden, channels_per_head=self.channels_per_head, expansion=self.expansion)


@dataclass_json
@dataclass
class TarflowConfig():
    channels_in:int
    """输入张量(图片)的通道数 `C`"""
    img_size:int
    """输入图像的边长 `W`"""
    patch_size:int
    """图像分块的边长 `P`"""

    channels_hidden:int
    """MetaBlock 中的隐藏层通道数 `C_hidden`"""
    blocks_num:int
    """MetaBlock 的数量"""
    layers_per_block:int
    """MetaBlock 中 AttentionBlock 的层数"""
    nvp:bool = True
    """是否使用 `NVP` 模式"""
    num_classes:int = 0
    """分类数量, 用于引导网络训练, 设置为 0 即为没有类别"""

    # MetaBlockConfig 的参数
    channels_per_head:int = 64
    """每个注意力头分配的通道数 `D`"""
    expansion = 4
    """MLP的隐藏层扩大系数."""

    @cached_property
    def num_patches(self) -> int:
        """图像被分割成的块数, 也即序列长度 `L`"""
        return (self.img_size // self.patch_size) ** 2

    @cached_property
    def channels_patched(self)->int:
        """输入张量被分块后的通道数, 也即每个块的通道数"""
        return self.channels_in * self.patch_size ** 2



    @cached_property
    def MetaBlockConfig(self,**kwargs) -> MetaBlockConfig:
        return MetaBlockConfig(
            channels_in=self.channels_patched,
            channels_hidden=self.channels_hidden,
            num_patches=self.num_patches,
            num_layers=self.layers_per_block,
            channels_per_head=self.channels_per_head,
            expansion= self.expansion,
            nvp=self.nvp,
            num_classes=self.num_classes
        )