from ....utils.code_utils import deprecate_module

deprecate_module("dataUtils", "data_utils", "0.16.0", error=True)

from .data_utils import *
