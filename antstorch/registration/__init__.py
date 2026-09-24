"""Registration components shared by ANTsTorch algorithms.

This namespace contains optimizer and field-update primitives that are not
specific to one registration model.  Higher-level implementations such as
SyN and DiReCT should depend on this package rather than on each other's
private modules.
"""

from .reg_adam import RegAdamState, reg_adam_direction

__all__ = [
    "RegAdamState",
    "reg_adam_direction",
]
