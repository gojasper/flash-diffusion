import logging

from .base import BaseFilter
from .filters_config import FilterOnConditionConfig, KeyFilterConfig

logging.basicConfig(level=logging.INFO)


class KeyFilter(BaseFilter):
    """
    This filter checks if ALL the given keys are present in the sample

    Args:

        config (KeyFilterConfig): configuration for the filter
    """

    def __init__(self, config: KeyFilterConfig):
        super().__init__(config)
        keys = config.keys
        if isinstance(keys, str):
            keys = [keys]

        self.keys = set(keys)

    def __call__(self, batch: dict) -> bool:
        try:
            res = self.keys.issubset(set(batch.keys()))
            return res
        except Exception as e:
            if self.verbose:
                logging.error(f"Error in KeyFilter: {e}")
            return False


class FilterOnCondition(BaseFilter):
    """
    This filter checks if the condition is satisfied

    Args:

        config (FilterOnConditionConfig): configuration for the filter
    """

    def __init__(self, config: FilterOnConditionConfig):
        super().__init__(config)
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.strict = config.strict

    def __call__(self, batch: dict) -> bool:
        if self.condition_key not in batch:
            if self.verbose:
                logging.error(
                    f"Error in FilterOnCondition: {self.condition_key} not in batch"
                )
            if self.strict:
                return False
            else:
                return True
        else:
            condition_key = batch[self.condition_key]
            return self.condition_fn(condition_key)
