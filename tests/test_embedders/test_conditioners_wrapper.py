import pytest
import torch

from flash.models.embedders import (
    ClipEmbedder,
    ClipEmbedderConfig,
    TorchNNEmbedder,
    TorchNNEmbedderConfig,
)
from flash.models.embedders.conditioners_wrapper import ConditionerWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestConditionerWrapper:
    @pytest.fixture(params=[0.0, 1.0])
    def unconditional_conditioning_rate(self, request):
        return request.param

    # Simmulate a config with a list of embedders
    @pytest.fixture(params=[0.0, 1.0])
    def embedder1(self, request):
        return ClipEmbedder(
            ClipEmbedderConfig(
                version="stabilityai/stable-diffusion-xl-base-1.0",
                text_embedder_subfolder="text_encoder",
                tokenizer_subfolder="tokenizer",
                input_key="text",
                unconditional_conditioning_rate=request.param,
            )
        )

    @pytest.fixture(params=[0.0, 1.0])
    def embedder2(self, request):
        return ClipEmbedder(
            ClipEmbedderConfig(
                version="stabilityai/stable-diffusion-xl-base-1.0",
                text_embedder_subfolder="text_encoder_2",
                tokenizer_subfolder="tokenizer_2",
                input_key="text",
                always_return_pooled=True,
                unconditional_conditioning_rate=request.param,
            )
        )

    @pytest.fixture(params=[0.0, 1.0])
    def embedder3(self, request):
        return TorchNNEmbedder(
            TorchNNEmbedderConfig(
                nn_modules=["torch.nn.Identity", "torch.nn.Conv2d"],
                nn_modules_kwargs=[
                    dict(),
                    dict(in_channels=3, out_channels=6, kernel_size=3, padding=1),
                ],
                input_key="image",
                unconditional_conditioning_rate=request.param,
            )
        )

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    @pytest.fixture()
    def conditioner_wrapper(self, embedder1, embedder2, embedder3):
        conditioner = ConditionerWrapper(conditioners=[embedder1, embedder2, embedder3])
        return conditioner

    @pytest.fixture(params=[["text", "image"], ["image"], [0]])
    def ucg_keys(self, request):
        return request.param

    def test_conditioner(
        self,
        conditioner_wrapper,
        conditioner_input,
        embedder1,
        embedder2,
        embedder3,
    ):
        conditioner_wrapper.to(DEVICE)
        output = conditioner_wrapper(conditioner_input, device=DEVICE)

        assert isinstance(output, dict)
        assert "cond" in output
        assert "crossattn" in output["cond"]
        assert "vector" in output["cond"]
        assert "concat" in output["cond"]

        if embedder1.ucg_rate == 1.0:
            assert torch.all(output["cond"]["crossattn"][:, :, :768] == 0.0)
        else:
            assert not torch.all(output["cond"]["crossattn"][:, :, :768] == 0.0)

        if embedder2.ucg_rate == 1.0:
            assert torch.all(output["cond"]["crossattn"][:, :, 768:] == 0.0)
            assert torch.all(output["cond"]["vector"] == 0.0)
        else:
            assert not torch.all(output["cond"]["crossattn"][:, :, 768:] == 0.0)
            assert not torch.all(output["cond"]["vector"] == 0.0)

        if embedder3.ucg_rate == 1.0:
            assert torch.all(output["cond"]["concat"] == 0.0)

        else:
            assert not torch.all(output["cond"]["concat"] == 0.0)

    def test_conditioner_with_ucg(
        self, conditioner_wrapper, conditioner_input, ucg_keys
    ):
        # Remove random conditioning
        for conditioner in conditioner_wrapper.conditioners:
            conditioner.ucg_rate = 0.0
        conditioner_wrapper.to(DEVICE)
        output = conditioner_wrapper(
            conditioner_input, ucg_keys=ucg_keys, device=DEVICE
        )
        assert isinstance(output, dict)
        assert "cond" in output
        assert "crossattn" in output["cond"]
        assert "vector" in output["cond"]
        assert "concat" in output["cond"]

        if "text" in ucg_keys:
            assert torch.all(output["cond"]["crossattn"] == 0.0)
            assert torch.all(output["cond"]["vector"] == 0.0)
        else:
            assert not torch.all(output["cond"]["crossattn"] == 0.0)
            assert not torch.all(output["cond"]["vector"] == 0.0)
        if "image" in ucg_keys:
            assert torch.all(output["cond"]["concat"] == 0.0)
        else:
            assert not torch.all(output["cond"]["concat"] == 0.0)
