from typing import Any, Callable, List, Optional, Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseFilterConfig(BaseConfig):
    """
    Base configuration for filters

    Args:

        verbose (bool):
            If True, print debug information. Defaults to False"""

    verbose: bool = False


@dataclass
class KeyFilterConfig(BaseFilterConfig):
    """
    This filter checks if the keys are present in a sample.

    Args:

        keys (Union[str, List[str]]):
            Key or list of keys to check. Defaults to "txt"
    """

    keys: Union[str, List[str]] = "txt"


@dataclass
class FilterOnConditionConfig(BaseFilterConfig):
    """
    This filter applies a filter based on a condition.

    Args:

        condition_key (Optional[str]): Key to use for the condition. Defaults to None
        condition_fn (Optional[Callable[[Any], bool]]): Function to use for the condition to be met so
            the sample is kept. Defaults to None.
        strict (bool): If True, the condition should be met for the sample to be kept. Defaults to False
    """

    condition_key: Optional[str] = None
    condition_fn: Optional[Callable[[Any], bool]] = None
    strict: bool = False
