import pytest
import torch

from flash.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestAutoencoderKLDiffusers:
    @pytest.fixture(
        params=[
            dict(),
            dict(
                version="stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="vae",
            ),
        ]
    )
    def model_config(self, request):
        return AutoencoderKLDiffusersConfig(
            **request.param, tiling_size=(16, 16), tiling_overlap=(8, 8), batch_size=1
        )

    @pytest.fixture()
    def model(self, model_config):
        return AutoencoderKLDiffusers(model_config).to(DEVICE)

    def test_model_initialization(self, model, model_config):
        assert model.config == model_config

    def test_encode(self, model):
        x = torch.randn(2, 3, 32, 32).to(DEVICE)
        z = model.encode(x)
        assert z.shape == (2, 4, 4, 4)

    def test_decode(self, model):
        z = torch.randn(2, 4, 4, 4).to(DEVICE)
        x = model.decode(z)
        assert x.shape == (2, 3, 32, 32)

    def test_decode_tiling(self, model):
        z = torch.randn(2, 4, 32, 32).to(DEVICE)
        x = model.decode(z)
        assert x.shape == (2, 3, 256, 256)
