import warnings

from theflow.utils.logging_utils import log_once


def warn_structure_refactor(old_module: str, new_module: str, direct: bool = True) -> None:
    """Create structure refactor warning to indicate modules new location post.

    Only creates a warning once per module.
    """
    old_module = old_module.replace(".py", "")
    if log_once(old_module):
        warning = (
            f"The module `{old_module}` has been moved to `{new_module}` and the old "
            f"location will be deprecated soon. Please adjust your imports to point "
            f"to the new location."
        )

        if direct:
            warning += f" Example: Do a global search and " f"replace `{old_module}` with `{new_module}`."
        else:
            warning += (
                f"\nATTENTION: This module may have been split or refactored. Please "
                f"check the contents of `{new_module}` before making changes."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(warning, DeprecationWarning, stacklevel=3)
