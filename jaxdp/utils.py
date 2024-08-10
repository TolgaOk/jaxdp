from typing import Callable, Union
from collections import defaultdict
import inspect
import functools


class StaticMeta(type):
    def __new__(cls, name, bases, attrs):
        # Make all methods static
        for key, value in attrs.items():
            if callable(value):
                attrs[key] = staticmethod(value)

        attrs["__doc__"] = f"This is the {name} class, which cannot be instantiated."

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"The {cls.__name__} class cannot be instantiated or called.")


def register_as(suffix: str) -> Callable:
    def register_suffix(function: Union[Callable, StaticMeta]) -> Union[Callable, StaticMeta]:
        fn_name = function.__name__
        caller_globals = inspect.stack()[1].frame.f_globals
        substitute_object = caller_globals.get(fn_name)

        if substitute_object is None:
            substitute_object = StaticMeta(fn_name, (), {
                suffix: function
            })
        elif isinstance(substitute_object, StaticMeta):
            setattr(substitute_object, suffix, function)
        else:
            raise ValueError(f"Function name: {fn_name} already exists!")

        return substitute_object
    return register_suffix
