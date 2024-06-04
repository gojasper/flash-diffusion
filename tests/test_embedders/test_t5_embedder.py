import pytest
import torch
from pydantic import ValidationError

from flash.models.embedders import T5TextEmbedder, T5TextEmbedderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestT5TextEmbedder:
    def test_wrong_config(self):
        with pytest.raises(ValidationError):
            T5TextEmbedderConfig(layer="hidden", layer_idx=None)

        with pytest.raises(ValidationError):
            T5TextEmbedderConfig(layer="hidden", layer_idx=25)

    @pytest.fixture(
        params=[
            [
                dict(
                    version="google/t5-v1_1-base",
                    tokenizer_max_length=None,
                    layer="hidden",
                    layer_idx=5,
                    returns_attention_mask=True,
                ),
                768,
            ],
            [
                dict(
                    version="PixArt-alpha/PixArt-XL-2-1024-MS",
                    text_embedder_subfolder="text_encoder",
                    tokenizer_subfolder="tokenizer",
                    tokenizer_max_length=120,
                ),
                4096,
            ],
        ]
    )
    def model_config_and_output_shape(self, request):
        return T5TextEmbedderConfig(**request.param[0]), request.param[1]

    @pytest.fixture()
    def model_config(self, model_config_and_output_shape):
        return model_config_and_output_shape[0]

    @pytest.fixture()
    def output_shape(self, model_config_and_output_shape):
        return model_config_and_output_shape[1]

    @pytest.fixture()
    def embedder(self, model_config):
        return T5TextEmbedder(model_config)

    @pytest.fixture(params=[True, False])
    def force_zero_embedding(self, request):
        return request.param

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    @torch.no_grad()
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
        if not model_config.returns_attention_mask:
            assert len(output) == 1
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                (
                    model_config.tokenizer_max_length
                    if model_config.tokenizer_max_length
                    else embedder.tokenizer.model_max_length
                ),
                output_shape,
            )

            if force_zero_embedding:
                assert torch.all(output["crossattn"] == 0)

        else:
            assert len(output) == 2
            assert output["crossattn"].shape == (
                len(conditioner_input[embedder.input_key]),
                (
                    model_config.tokenizer_max_length
                    if model_config.tokenizer_max_length
                    else embedder.tokenizer.model_max_length
                ),
                output_shape,
            )
            assert output["attention_mask"].shape == (
                len(conditioner_input[embedder.input_key]),
                (
                    model_config.tokenizer_max_length
                    if model_config.tokenizer_max_length
                    else embedder.tokenizer.model_max_length
                ),
            )

            if force_zero_embedding:
                assert torch.all(output["crossattn"] == 0)
                assert torch.all(output["attention_mask"] == 0)
