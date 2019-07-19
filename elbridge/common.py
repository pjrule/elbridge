""" Common utility functions for Elbridge. """
from typing import Any

# NOTE: Having a large collection of common utility functions is generally
# considered an antipattern, as it makes separation of concerns fuzzier.
# A utility function should only be added here if:
#  a. It is short (preferably <15 lines) and relatively generic.
#  b. It is used by several modules and thus would significantly impede
#     maintainability if duplicated.
# c. Placing it in a more specific module would create
#    an icky circular dependency.


def bound(val: Any, min_val: Any, max_val: Any) -> Any:
    """ Bounds a value between a maximum and a minimum. """
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    return val
