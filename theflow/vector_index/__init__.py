import logging
from typing import Type

from theflow.api_annotations import DeveloperAPI
from theflow.vector_index.base import VectorIndex

logger = logging.getLogger(__name__)


FAISS = "faiss"

ALL_INDICES = [FAISS]


def get_faiss_index_cls() -> Type[VectorIndex]:
    from theflow.vector_index.faiss import FaissIndex

    return FaissIndex


# TODO(travis): add other indexing structures
vector_index_registry = {
    FAISS: get_faiss_index_cls,
}


@DeveloperAPI
def get_vector_index_cls(type: str) -> Type[VectorIndex]:
    return vector_index_registry[type]()
