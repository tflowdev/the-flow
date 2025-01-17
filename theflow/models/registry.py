import logging

from theflow.constants import MODEL_ECD, MODEL_GBM, MODEL_LLM
from theflow.models.ecd import ECD
from theflow.models.llm import LLM

logger = logging.getLogger(__name__)


def gbm(*args, **kwargs):
    try:
        from theflow.models.gbm import GBM
    except ImportError:
        logger.warning(
            "Importing GBM requirements failed. Not loading GBM model type. "
            "If you want to use GBM, install The Flow's 'tree' extra."
        )
        raise

    return GBM(*args, **kwargs)


model_type_registry = {
    MODEL_ECD: ECD,
    MODEL_GBM: gbm,
    MODEL_LLM: LLM,
}
