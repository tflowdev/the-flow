from theflow.api_annotations import DeveloperAPI
from theflow.constants import AUGMENTATION, IMAGE, TYPE
from theflow.schema import utils as schema_utils
from theflow.schema.features.augmentation.base import BaseAugmentationConfig
from theflow.schema.features.augmentation.utils import register_augmentation_config
from theflow.schema.metadata import FEATURE_METADATA
from theflow.schema.utils import theflow_dataclass


@DeveloperAPI
@register_augmentation_config(name="auto_augmentation", features=IMAGE)
@theflow_dataclass
class AutoAugmentationConfig(BaseAugmentationConfig):
    """Automatic augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "auto_augmentation",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )
    method: str = schema_utils.String(
        default="trivial_augment",
        description="Specifies the method for applying automatic data augmentation",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["auto_augmentation_method"],
    )


@DeveloperAPI
@register_augmentation_config(name="random_horizontal_flip", features=IMAGE)
@theflow_dataclass
class RandomHorizontalFlipConfig(BaseAugmentationConfig):
    """Random horizontal flip augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_horizontal_flip",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )


@DeveloperAPI
@register_augmentation_config(name="random_vertical_flip", features=IMAGE)
@theflow_dataclass
class RandomVerticalFlipConfig(BaseAugmentationConfig):
    """Random vertical flip augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_vertical_flip",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )


@DeveloperAPI
@register_augmentation_config(name="random_rotate", features=IMAGE)
@theflow_dataclass
class RandomRotateConfig(BaseAugmentationConfig):
    """Random rotation augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_rotate",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["type"],
    )
    degree: int = schema_utils.Integer(
        default=15,
        description="Range of angle for random rotation, i.e.,  [-degree, +degree].",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["rotation_degree"],
    )


@DeveloperAPI
@register_augmentation_config(name="random_blur", features=IMAGE)
@theflow_dataclass
class RandomBlurConfig(BaseAugmentationConfig):
    """Random blur augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_blur",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )
    kernel_size: int = schema_utils.Integer(
        default=3,
        description="Kernel size for random blur.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["kernel_size"],
    )


@DeveloperAPI
@register_augmentation_config(name="random_brightness", features=IMAGE)
@theflow_dataclass
class RandomBrightnessConfig(BaseAugmentationConfig):
    """Random brightness augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_brightness",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )

    min: int = schema_utils.FloatRange(
        default=0.5,
        description="Minimum factor for random brightness.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["min_brightness"],
    )

    max: int = schema_utils.FloatRange(
        default=2.0,
        description="Maximum factor for random brightness.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["max_brightness"],
    )


@DeveloperAPI
@register_augmentation_config(name="random_contrast", features=IMAGE)
@theflow_dataclass
class RandomContrastConfig(BaseAugmentationConfig):
    """Random Contrast augmentation operation."""

    type: str = schema_utils.ProtectedString(
        "random_contrast",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION][TYPE],
    )

    min: int = schema_utils.FloatRange(
        default=0.5,
        description="Minimum factor for random brightness.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["min_contrast"],
    )

    max: int = schema_utils.FloatRange(
        default=2.0,
        description="Maximum factor for random brightness.",
        parameter_metadata=FEATURE_METADATA[IMAGE][AUGMENTATION]["max_contrast"],
    )
