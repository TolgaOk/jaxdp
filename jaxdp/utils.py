from typing import Callable

class StaticMeta(type):
    def __new__(cls, name, bases, attrs):

        included_method_names = []
        for key, value in attrs.items():
            if isinstance(value, staticmethod):
                raise ValueError(f"staticmethod is not allowed! Method: {value}")
            if callable(value):
                attrs[key] = value
            if type(value) is StaticMeta:
                getattr(value, "__inherited_names").insert(0, name)
                for sub_method_name in getattr(value, "__included_method_names"):
                    included_method_names.append(".".join([key, sub_method_name]))
            elif callable(value):
                included_method_names.append(key)

        options_string = ", ".join(f"{fn_name}" for fn_name in included_method_names)
        attrs["__doc__"] = f"Main object for {name}. Usable attributes are {options_string}"
        attrs["__inherited_names"] = [name]
        attrs["__included_method_names"] = included_method_names

        def _init_method_(self, *args, **kwargs):
            main_name = ".".join(getattr(self, "__inherited_names"))
            options_string = " or ".join(
                f"{main_name}.{fn_name}" for fn_name in included_method_names)
            raise AttributeError(
                f"Attribute required! Call {options_string}, instead of {name}.")

        attrs["__init__"] = _init_method_

        return super().__new__(cls, name, bases, attrs)
