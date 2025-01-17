# register trainers

import theflow.trainers.trainer  # noqa: F401

try:
    import theflow.trainers.trainer_lightgbm  # noqa: F401
except ImportError:
    pass


try:
    import theflow.trainers.trainer_llm  # noqa: F401
except ImportError:
    pass
