import json
import logging
from typing import Any, Dict

from controlnet_aux import CannyDetector, MidasDetector
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from .base import BaseMapper
from .mappers_config import (
    CannyEdgeMapperConfig,
    KeyRenameMapperConfig,
    KeysFromJSONMapperConfig,
    MidasDepthMapperConfig,
    RemoveKeysMapperConfig,
    RescaleMapperConfig,
    SelectKeysMapperConfig,
    SetValueConfig,
    TorchvisionMapperConfig,
)


class KeyRenameMapper(BaseMapper):
    """
    Rename keys in a sample according to a key map

    Args:

        config (KeyRenameMapperConfig): Configuration for the mapper

    Examples
    ########

    1. Rename keys in a sample according to a key map

    .. code-block:: python

        from flash.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"}
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

    2. Rename keys in a sample according to a key map and a condition key

    .. code-block:: python

        from flash.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"},
            condition_key="condition",
            condition_fn=lambda x: x == 1
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1, "condition": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

        sample = {"old_key": 1, "condition": 0}
        new_sample = mapper(sample)
        print(new_sample)  # {"old_key": 1}

    ```
    """

    def __init__(self, config: KeyRenameMapperConfig):
        super().__init__(config)
        self.key_map = config.key_map
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.else_key_map = config.else_key_map

    def __call__(self, batch: dict):
        if self.condition_key is not None:
            condition_key = batch[self.condition_key]
            if self.condition_fn(condition_key):
                for old_key, new_key in self.key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

            elif self.else_key_map is not None:
                for old_key, new_key in self.else_key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

        else:
            for old_key, new_key in self.key_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)
        return batch


class TorchvisionMapper(BaseMapper):
    """
    Apply torchvision transforms to a sample

    Args:

        config (TorchvisionMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: TorchvisionMapperConfig):
        super().__init__(config)
        chained_transforms = []
        for transform, kwargs in zip(config.transforms, config.transforms_kwargs):
            transform = getattr(transforms, transform)
            chained_transforms.append(transform(**kwargs))
        self.transforms = transforms.Compose(chained_transforms)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch[self.output_key] = self.transforms(batch[self.key])
        return batch


class RescaleMapper(BaseMapper):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        config (RescaleMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RescaleMapperConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(batch[self.key], list):
            tmp = []
            for i, image in enumerate(batch[self.key]):
                tmp.append(2 * image - 1)
            batch[self.output_key] = tmp
        else:
            batch[self.output_key] = 2 * batch[self.key] - 1
        return batch


class KeysFromJSONMapper(BaseMapper):
    """
    Get keys from a JSON string and add them to the batch

    Args:

        config (KeysFromJSONMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: KeysFromJSONMapperConfig):
        super().__init__(config)
        keys_to_extract = config.keys_to_extract
        self.remove_original = config.remove_original
        self.strict = config.strict

        if isinstance(keys_to_extract, str):
            self.keys_to_extract = [keys_to_extract]
        else:
            self.keys_to_extract = keys_to_extract

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        assert self.key in batch, f"Key {self.key} not in batch"
        if isinstance(batch[self.key], str):
            decoded_json = json.loads(batch[self.key])
        elif isinstance(batch[self.key], dict):
            decoded_json = batch[self.key]

        for key in self.keys_to_extract:
            try:
                batch[key] = decoded_json[key]
            except KeyError as e:
                # If the key is not found, raise an error or continue
                if self.strict:
                    raise e
                # If the key is not found, continue
                else:
                    logging.debug(f"Key {key} not found in JSON")
                    continue
        if self.remove_original:
            del batch[self.key]
        return batch


class SelectKeysMapper(BaseMapper):
    """
    Select keys from a sample and remove the rest

    Args:

        config (SelectKeysMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: SelectKeysMapperConfig):
        super().__init__(config)
        keys = config.keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {key: batch[key] for key in self.keys}


class RemoveKeysMapper(BaseMapper):
    """
    Remove keys from a sample

    Args:

        config (RemoveKeysMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RemoveKeysMapperConfig):
        super().__init__(config)
        keys = config.keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.keys:
            del batch[key]
        return batch


class SetValueMapper(BaseMapper):
    """SetValueMapper

    Set a value to a key in the batch

    Args:
        config (SetValueConfig): Configuration for the mapper
    """

    def __init__(self, config: SetValueConfig):
        super().__init__(config)
        self.value = config.value
        self.key = config.key

    def __call__(self, batch: dict):
        batch[self.key] = self.value
        return batch


class CannyEdgeMapper(BaseMapper):
    """CannyEdgeMapper

    Apply Canny edge detection to an image

    Args:
        config (CannyEdgeConfig): Configuration for the mapper
    """

    def __init__(self, config: CannyEdgeMapperConfig):
        super().__init__(config)
        self.image_key = config.key
        self.output_key = config.output_key
        self.detect_resolution = config.detect_resolution
        self.image_resolution = config.image_resolution
        self.mode = config.mode
        self.canny_detector = CannyDetector()

    def __call__(self, batch: dict):
        image = batch[self.image_key]
        assert isinstance(image, Image.Image)
        edges = self.canny_detector(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
        )
        edges = edges.convert(self.mode)
        batch[self.output_key] = edges
        return batch


class MidasDepthMapper(BaseMapper):
    """MidasDepthMapper

    Apply Midas depth estimation to an image

    Args:
        config (MidasDepthMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: MidasDepthMapperConfig):
        super().__init__(config)
        self.image_key = config.key
        self.output_key = config.output_key
        self.detect_resolution = config.detect_resolution
        self.image_resolution = config.image_resolution
        self.mode = config.mode
        self.midas_detector = MidasDetector.from_pretrained(
            "valhalla/t2iadapter-aux-models",
            filename="dpt_large_384.pt",
            model_type="dpt_large",
        )

    def __call__(self, batch: dict):
        image = batch[self.image_key]
        assert isinstance(image, Image.Image)
        depth = self.midas_detector(
            image,
            detect_resolution=self.detect_resolution,
            image_resolution=self.image_resolution,
        )
        depth = depth.convert(self.mode)
        batch[self.output_key] = depth
        return batch
