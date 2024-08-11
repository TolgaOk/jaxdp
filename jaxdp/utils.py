from typing import Callable
import sys


class StaticMeta(type):
    def __new__(cls, name, bases, attrs):

        for key, value in attrs.items():
            if callable(value):
                attrs[key] = staticmethod(value)

        attrs["__doc__"] = f"This is the {name} class, which cannot be instantiated."

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"The {cls.__name__} class cannot be instantiated or called.")


def register_as(suffix: str) -> Callable:
    def register_suffix(function: Callable) -> Callable:
        fn_name = function.__qualname__

        module_name = function.__module__
        module = sys.modules[module_name]

        module_globals = module.__dict__
        substitute_object = module_globals.get(fn_name)

        if substitute_object is None:
            substitute_object = StaticMeta(fn_name, (), {
                suffix: function,
                "__module__": module_name
            })
            module_globals[fn_name] = substitute_object
        elif isinstance(substitute_object, StaticMeta):
            setattr(substitute_object, suffix, function)
        else:
            raise ValueError(f"Function name: {fn_name} already exists!")

        return substitute_object

    return register_suffix
