from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseMapperConfig(BaseConfig):
    """
    Base configuration for mappers.

    Args:

        verbose (bool):
            If True, print debug information. Defaults to False

        key (Optional[str]):
            Key to apply the mapper to. Defaults to None

        output_key (Optional[str]):
            Key to store the output of the mapper. Defaults to None
    """

    verbose: bool = False
    key: Optional[str] = None
    output_key: Optional[str] = None


@dataclass
class KeyRenameMapperConfig(BaseMapperConfig):
    """
    Rename keys in a sample according to a key map

    Args:

        key_map (Dict[str, str]): Dictionary with the old keys as keys and the new keys as values
        condition_key (Optional[str]): Key to use for the condition. Defaults to None
        condition_fn (Optional[Callable[[Any], bool]]): Function to use for the condition to be met so
            the key map is applied. Defaults to None.
        else_key_map (Optional[Dict[str, str]]): Dictionary with the old keys as keys and the new keys as values
            if the condition is not met. Defaults to None *i.e.* the original key will be used.
    """

    key_map: Dict[str, str] = None
    condition_key: Optional[str] = None
    condition_fn: Optional[Callable[[Any], bool]] = None
    else_key_map: Optional[Dict[str, str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.key_map is not None, "key_map should be provided"
        assert all(
            isinstance(old_key, str) and isinstance(new_key, str)
            for old_key, new_key in self.key_map.items()
        ), "key_map should be a dictionary with string keys and values"
        if self.condition_key is not None:
            assert self.condition_fn is not None, "condition_fn should be provided"
            assert callable(self.condition_fn), "condition_fn should be callable"
        if self.condition_fn is not None:
            assert self.condition_key is not None, "condition_key should be provided"
            assert isinstance(
                self.condition_key, str
            ), "condition_key should be a string"
        if self.else_key_map is not None:
            assert all(
                isinstance(old_key, str) and isinstance(new_key, str)
                for old_key, new_key in self.else_key_map.items()
            ), "else_key_map should be a dictionary with string keys and values"


@dataclass
class TorchvisionMapperConfig(BaseMapperConfig):
    """
    Apply torchvision transforms to a sample

    Args:

        key (str): Key to apply the transforms to
        transforms (torchvision.transforms): List of torchvision transforms to apply
        transforms_kwargs (Dict[str, Any]): List of kwargs for the transforms
    """

    key: str = "image"
    transforms: List[str] = None
    transforms_kwargs: List[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.transforms is None:
            self.transforms = []
        if self.transforms_kwargs is None:
            self.transforms_kwargs = []
        assert len(self.transforms) == len(
            self.transforms_kwargs
        ), "Number of transforms and kwargs should be same"


@dataclass
class RescaleMapperConfig(BaseMapperConfig):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        key (str): Key to rescale
    """

    key: str = "image"


@dataclass
class KeysFromJSONMapperConfig(BaseMapperConfig):
    """
    Get keys from a JSON string and add them to the batch

    Args:

        key (str): Key to extract from the batch
        keys_to_extract (Union[str, List[str]]): Keys to extract from the JSON string
        remove_original (bool): Whether to remove the original key from the batch
        strict (bool): Whether to raise an error if a key is not found in the JSON string
    """

    key: str = "json"
    keys_to_extract: Union[str, List[str]] = None
    remove_original: bool = True
    strict: bool = True


@dataclass
class SelectKeysMapperConfig(BaseMapperConfig):
    """
    Selects keys from the batch

    Args:

        keys (Union[str, List[str]]): Keys to select
    """

    keys: Union[str, List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.keys is not None, "keys should be provided"


@dataclass
class RemoveKeysMapperConfig(BaseMapperConfig):
    """
    Removes keys from the batch

    Args:

        keys (Union[str, List[str]]): Keys to remove
    """

    keys: Union[str, List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.keys is not None, "keys should be provided"


@dataclass
class SetValueConfig(BaseMapperConfig):
    """
    Set a value in the batch

    Args:

        key (str): Key to apply the mapper to
        value (Any): Value to set
    """

    value: Any = None


@dataclass
class CannyEdgeMapperConfig(BaseMapperConfig):
    """
    Apply Canny edge detection to an image

    Args:

        output_key (str): Key to store the output of the mapper
        detect_resolution (int): Resolution to detect the edges
        image_resolution (int): Resolution of the output edge image
        mode (str): Mode of the output image
    """

    key: str = "image"
    output_key: str = "edges"
    detect_resolution: int = 384
    image_resolution: int = 1024
    mode: Literal["L", "RGB"] = "RGB"


@dataclass
class MidasDepthMapperConfig(BaseMapperConfig):
    """
    Apply depthmap to an image

    Args:

        output_key (str): Key to store the output of the mapper
        detect_resolution (int): Resolution to detect the edges
        image_resolution (int): Resolution of the output edge image
        mode (str): Mode of the output image
    """

    key: str = "image"
    output_key: str = "depth"
    detect_resolution: int = 512
    image_resolution: int = 1024
    mode: Literal["L", "RGB"] = "RGB"
