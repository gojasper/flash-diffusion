from typing import Any, Dict

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from ..base import BaseConditioner
from .clip_embedder_config import ClipEmbedderConfig


class ClipEmbedder(BaseConditioner):
    """This is the ClipEmbedder class which defines the ClipEmbedder model

    Args:

        config (ClipEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: ClipEmbedderConfig):
        BaseConditioner.__init__(self, config)

        # overides predefined configuration
        kwargs = dict()
        if config.pad_token is not None:
            kwargs["pad_token"] = config.pad_token

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.version,
            subfolder=config.tokenizer_subfolder,
            revision=config.tokenizer_revision,
            **kwargs
        )
        self.transformer = CLIPTextModel.from_pretrained(
            config.version,
            subfolder=config.text_embedder_subfolder,
            revision=config.text_embedder_revision,
        )
        self.max_length = self.tokenizer.model_max_length
        self.layer = config.layer
        self.layer_idx = config.layer_idx
        self.always_return_pooled = config.always_return_pooled
        self.tokenizer_truncation = config.tokenizer_truncation
        self.tokenizer_return_length = config.tokenizer_return_length

    def freeze(self):
        super().freeze()
        self.transformer = self.transformer.eval()

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device="cpu",
        *args,
        **kwargs
    ):
        """
        Forward pass of the ClipEmbedder

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
        )
        tokens = batch_encoding["input_ids"].to(device)
        self.transformer = self.transformer.to(device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]

        if force_zero_embedding:
            z = 0 * z
        output = {self.dim2outputkey[z.dim()]: z}
        if self.always_return_pooled:
            if force_zero_embedding:
                outputs.pooler_output = 0 * outputs.pooler_output
            output.update({self.dim2outputkey[2]: outputs.pooler_output})
        return output


class ClipEmbedderWithProjection(BaseConditioner):
    """
    ClipEmbedderWithProjection class which defines the ClipEmbedderWithProjection model

    Args:

        config (ClipEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: ClipEmbedderConfig):
        BaseConditioner.__init__(self, config)

        # overides predefined configuration
        kwargs = dict()
        if config.pad_token is not None:
            kwargs["pad_token"] = config.pad_token

        # hack to fix the projection dim for the L versions of laion
        if config.version in [
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        ]:
            kwargs["projection_dim"] = 768

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config.version,
            subfolder=config.tokenizer_subfolder,
            revision=config.tokenizer_revision,
            **kwargs
        )
        self.transformer = CLIPTextModelWithProjection.from_pretrained(
            config.version,
            subfolder=config.text_embedder_subfolder,
            revision=config.text_embedder_revision,
        )
        self.max_length = self.tokenizer.model_max_length
        self.layer = config.layer
        self.layer_idx = config.layer_idx
        self.always_return_pooled = config.always_return_pooled
        self.tokenizer_truncation = config.tokenizer_truncation
        self.tokenizer_return_length = config.tokenizer_return_length

    def freeze(self):
        super().freeze()
        self.transformer = self.transformer.eval()

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        device="cpu",
        *args,
        **kwargs
    ):
        """
        Forward pass of the ClipEmbedderWithProjection

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
        )
        tokens = batch_encoding["input_ids"].to(device)
        self.transformer = self.transformer.to(device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.text_embeds[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]

        if force_zero_embedding:
            z = 0 * z
        output = {self.dim2outputkey[z.dim()]: z}
        if self.always_return_pooled:
            if force_zero_embedding:
                outputs.text_embeds = 0 * outputs.text_embeds
            output.update({self.dim2outputkey[2]: outputs.text_embeds})
        return output
