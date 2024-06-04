import pytest
import torch
from pydantic import ValidationError

from flash.models.embedders import TorchNNEmbedder, TorchNNEmbedderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestTorchNNEmbedderInstanciation:
    def test_raises_len_mismatch(self):
        with pytest.raises(ValidationError):
            TorchNNEmbedderConfig(
                nn_modules=["torch.nn.Identity", "torch.nn.Linear"],
                nn_modules_kwargs=[{"in_features": 3, "out_features": 3}],
            )

    @pytest.fixture(
        params=[
            [
                ["torch.nn.Identity", "torch.nn.Linear"],
                [dict(), dict(in_features=12, out_features=24)],
            ],
            [
                ["torch.nn.Conv2d", "torch.nn.Conv2d"],
                [
                    dict(in_channels=3, out_channels=6, kernel_size=3, padding=1),
                    dict(in_channels=6, out_channels=12, kernel_size=3, padding=1),
                ],
            ],
        ]
    )
    def config(self, request):
        return TorchNNEmbedderConfig(
            nn_modules=request.param[0], nn_modules_kwargs=request.param[1]
        )

    def test_instanciate(self, config):
        embedder = TorchNNEmbedder(config)

        for i in range(len(config.nn_modules)):
            assert isinstance(embedder.nn_modules[i], eval(config.nn_modules[i]))


class TestTorchNNEmbedderForward:
    @pytest.fixture(
        params=[
            [
                ["torch.nn.Conv2d", "torch.nn.Conv2d"],
                [
                    dict(in_channels=3, out_channels=6, kernel_size=3, padding=1),
                    dict(in_channels=6, out_channels=12, kernel_size=3, padding=1),
                ],
            ],
        ]
    )
    def config(self, request):
        return TorchNNEmbedderConfig(
            nn_modules=request.param[0],
            nn_modules_kwargs=request.param[1],
            input_key="image",
        )

    @pytest.fixture(params=[True, False])
    def force_zero_embedding(self, request):
        return request.param

    @pytest.fixture()
    def embedder(self, config):
        return TorchNNEmbedder(config)

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    def test_conditioner_forward(
        self, embedder, conditioner_input, force_zero_embedding
    ):
        embedder.to(DEVICE)
        output = embedder(conditioner_input, force_zero_embedding=force_zero_embedding)

        if list(embedder.nn_modules) != []:
            assert output["concat"].shape == (2, 12, 32, 32)
        else:
            assert output["concat"].shape == conditioner_input[embedder.input_key].shape

        if force_zero_embedding:
            assert torch.all(output["concat"] == 0.0)
