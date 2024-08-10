from typing import NewType, Union
import jax
from jaxtyping import Array, Float32, Int32, Bool


class F(type):
    def __class_getitem__(cls, dim_str: str):
        type_var = Union[jax.core.Tracer, Float32[Array, dim_str]]
        return NewType(f"{cls.__name__}[{dim_str}]", type_var)


class I(type):
    def __class_getitem__(cls, dim_str: str):
        type_var = Union[jax.core.Tracer, Int32[Array, dim_str]]
        return NewType(f"{cls.__name__}[{dim_str}]", type_var)


class B(type):
    def __class_getitem__(cls, dim_str: str):
        type_var = Union[jax.core.Tracer, Bool[Array, dim_str]]
        return NewType(f"{cls.__name__}[{dim_str}]", type_var)


QType = NewType("QType", F["A S"])
VType = NewType("VType", F["S"])
PiType = NewType("PiType", F["A S"])
