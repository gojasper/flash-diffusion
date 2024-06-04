from typing import Any, Dict

from transformers import T5EncoderModel, T5Tokenizer

from flash.models.embedders.base.base_conditioner_config import BaseConditionerConfig

from ..base import BaseConditioner
from .t5_embedder_config import T5TextEmbedderConfig


class T5TextEmbedder(BaseConditioner):
    """
    This is the T5TextEmbedder class which defines the T5TextEmbedder model

    Args:
        config (T5TextEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: T5TextEmbedderConfig):
        BaseConditioner.__init__(self, config)

        self.tokenizer = T5Tokenizer.from_pretrained(
            config.version,
            subfolder=config.tokenizer_subfolder,
            revision=config.tokenizer_revision,
        )

        self.transformer = T5EncoderModel.from_pretrained(
            config.version,
            subfolder=config.text_embedder_subfolder,
            revision=config.text_embedder_revision,
        )
        if config.tokenizer_max_length is not None:
            self.max_length = config.tokenizer_max_length
        else:
            self.max_length = self.tokenizer.model_max_length
        self.layer = config.layer
        self.layer_idx = config.layer_idx
        self.returns_attention_mask = config.returns_attention_mask
        self.tokenizer_truncation = config.tokenizer_truncation
        self.tokenizer_return_length = config.tokenizer_return_length
        self.tokenizer_add_special_tokens = config.tokenizer_add_special_tokens

    def freeze(self):
        super().freeze()
        self.transformer = self.transformer.eval()

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device: str = "cpu",
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass of the T5TextEmbedder

        Args:

            batch (Dict[str, Any]): The batch of data
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.
            device (str): The device to use. Defaults to "cpu".

        Returns:

            Dict[str, Any]: The output of the embedder. This embedder outputs a 2-dimensional conditioning (type "crossattn")
                and a 1-dimensional conditioning (type "vector") if always_return_pooled is True.
        """
        text = batch[self.input_key]
        batch_encoding = self.tokenizer(
            text,
            truncation=self.tokenizer_truncation,
            max_length=self.max_length,
            return_length=self.tokenizer_return_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=self.tokenizer_add_special_tokens,
        )
        tokens = batch_encoding["input_ids"].to(device)
        attention_mask = batch_encoding["attention_mask"].to(device)
        self.transformer = self.transformer.to(device)
        outputs = self.transformer(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=self.layer == "hidden",
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.layer_idx]

        if force_zero_embedding:
            z = 0 * z
            attention_mask = 0 * attention_mask

        if self.returns_attention_mask:
            output = {self.dim2outputkey[z.dim()]: z, "attention_mask": attention_mask}
        else:
            output = {self.dim2outputkey[z.dim()]: z}

        return output
