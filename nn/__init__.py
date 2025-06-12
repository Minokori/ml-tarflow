#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


from dataclasses import dataclass
from typing import Literal, NamedTuple


@dataclass
class ReverseHyperParameters():
    guidance: float = 0.0
    """类别引导权重. `0` 表示无条件指导"""
    guide_what: Literal["", "a", "b", "ab"] = "ab"
    """类别引导的内容.

    + `a` 表示引导 α(x)
    + `b` 表示引导 μ(x)
    """
    tau: float = 1.0
    """手动注入温度项."""
    annealed_guidance: bool = False
    """是否使用退火类别引导. `True` 表示使用, `False` 表示不使用."""

    # GSJ paper 引入的超参数

    num_GS: int = 1

    max_jacobi: int = 100
    """最大 Jacobi 迭代次数."""
    zero_guess: int = 0
    ebound: float = 1e-8

    @property
    def no_classification_guide(self) -> bool:
        """是否需要类别引导."""
        return self.guidance > 0.0 and self.guide_what != ""
