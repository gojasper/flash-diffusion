import pytest
import torch
from pydantic import ValidationError

from flash.models.embedders import (
    ClipEmbedder,
    ClipEmbedderConfig,
    ClipEmbedderWithProjection,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestClipEmbedder:
    def test_wrong_config(self):
        with pytest.raises(ValidationError):
            ClipEmbedderConfig(layer="hidden", layer_idx=None)

        with pytest.raises(ValidationError):
            ClipEmbedderConfig(layer="hidden", layer_idx=13)

    @pytest.fixture(
        params=[
            [
                dict(
                    always_return_pooled=True,
                ),
                768,
            ],
            [
                dict(
                    version="stabilityai/stable-diffusion-xl-base-1.0",
                    text_embedder_subfolder="text_encoder",
                    tokenizer_subfolder="tokenizer",
                ),
                768,
            ],
        ]
    )
    def model_config_and_output_shape(self, request):
        return ClipEmbedderConfig(**request.param[0]), request.param[1]

    @pytest.fixture()
    def model_config(self, model_config_and_output_shape):
        return model_config_and_output_shape[0]

    @pytest.fixture()
    def output_shape(self, model_config_and_output_shape):
        return model_config_and_output_shape[1]

    @pytest.fixture()
    def embedder(self, model_config):
        return ClipEmbedder(model_config)

    @pytest.fixture(params=[True, False])
    def force_zero_embedding(self, request):
        return request.param

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    def test_model_forward(
        self,
        model_config,
        embedder,
        conditioner_input,
        output_shape,
        force_zero_embedding,
    ):
        embedder.to(DEVICE)
        output = embedder(
            conditioner_input, force_zero_embedding=force_zero_embedding, device=DEVICE
        )
        if model_config.always_return_pooled:
            assert len(output) == 2
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                embedder.tokenizer.model_max_length,
                output_shape,
            )
            assert output["vector"].shape == (
                len(conditioner_input[embedder.input_key]),
                output_shape,
            )

        else:
            assert len(output) == 1
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                embedder.tokenizer.model_max_length,
                output_shape,
            )

        if force_zero_embedding:
            assert torch.all(output["crossattn"] == 0)
            if model_config.always_return_pooled:
                assert torch.all(output["vector"] == 0)


class TestClipEmbedderWithProjection:
    def test_wrong_config(self):
        with pytest.raises(ValidationError):
            ClipEmbedderConfig(layer="hidden", layer_idx=None)

        with pytest.raises(ValidationError):
            ClipEmbedderConfig(layer="hidden", layer_idx=13)

    @pytest.fixture(
        params=[
            [
                dict(
                    version="stabilityai/stable-diffusion-xl-base-1.0",
                    text_embedder_subfolder="text_encoder_2",
                    tokenizer_subfolder="tokenizer_2",
                    always_return_pooled=True,
                ),
                1280,
            ],
            [
                dict(
                    version="stabilityai/stable-diffusion-xl-base-1.0",
                    text_embedder_subfolder="text_encoder_2",
                    tokenizer_subfolder="tokenizer_2",
                ),
                1280,
            ],
        ]
    )
    def model_config_and_output_shape(self, request):
        return ClipEmbedderConfig(**request.param[0]), request.param[1]

    @pytest.fixture()
    def model_config(self, model_config_and_output_shape):
        return model_config_and_output_shape[0]

    @pytest.fixture()
    def output_shape(self, model_config_and_output_shape):
        return model_config_and_output_shape[1]

    @pytest.fixture()
    def embedder(self, model_config):
        return ClipEmbedderWithProjection(model_config)

    @pytest.fixture(params=[True, False])
    def force_zero_embedding(self, request):
        return request.param

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    def test_model_forward(
        self,
        model_config,
        embedder,
        conditioner_input,
        output_shape,
        force_zero_embedding,
    ):
        embedder.to(DEVICE)
        output = embedder(
            conditioner_input, force_zero_embedding=force_zero_embedding, device=DEVICE
        )
        if model_config.always_return_pooled:
            assert len(output) == 2
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                embedder.tokenizer.model_max_length,
                output_shape,
            )
            assert output["vector"].shape == (
                len(conditioner_input[embedder.input_key]),
                output_shape,
            )

        else:
            assert len(output) == 1
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                embedder.tokenizer.model_max_length,
                output_shape,
            )

        if force_zero_embedding:
            assert torch.all(output["crossattn"] == 0)
            if model_config.always_return_pooled:
                assert torch.all(output["vector"] == 0)
