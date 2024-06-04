import pytest
import torch

from flash.models.unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDiffusersUNet2DWrapper:
    # simulates class conditioning
    @pytest.fixture(params=[None, torch.randint(256, (2,)).to(DEVICE)])
    def conditioning(self, request):
        if request.param is not None:
            return {"cond": {"vector": request.param}}
        return None

    # simulates a latent sample
    @pytest.fixture()
    def sample(self):
        return torch.rand(2, 6, 32, 32).to(DEVICE)

    # simulates a timestep
    @pytest.fixture(
        params=[10.0, torch.randint(1000, (2,), dtype=torch.float).to(DEVICE), 3]
    )
    def timesteps(self, request):
        return request.param

    def test_unet2d_wrapper(self, sample, timesteps, conditioning):
        unet = DiffusersUNet2DWrapper(
            sample_size=sample.shape[2:],
            in_channels=sample.shape[1],
            out_channels=3,
            num_class_embeds=256 if conditioning else None,
        ).to(DEVICE)
        output = unet(sample, timesteps, conditioning)
        assert output.shape == (
            sample.shape[0],
            3,
            sample.shape[2],
            sample.shape[3],
        )


class TestDiffusersUNet2DCondWrapper:
    # simulates class conditioning
    @pytest.fixture(params=[None, torch.randn(2, 256).to(DEVICE)])
    def vector_conditioning(self, request):
        if request.param is not None:
            return {"vector": request.param}
        return None

    # simulates crossattn conditioning '(always needed for conditional UNet2D)' (see diffusers/models/unet.py
    @pytest.fixture()
    def crossattn_conditioning(self):
        return {"crossattn": torch.randn(2, 12, 123).to(DEVICE)}

    # simulates concat conditioning
    @pytest.fixture(params=[None, torch.randn(2, 2, 32, 32).to(DEVICE)])
    def concat_conditioning(self, request):
        if request.param is not None:
            return {"concat": request.param}
        return None

    @pytest.fixture()
    def conditioning(
        self, vector_conditioning, crossattn_conditioning, concat_conditioning
    ):
        cond = dict(cond=crossattn_conditioning)
        if vector_conditioning is not None:
            cond["cond"].update(vector_conditioning)
        if concat_conditioning is not None:
            cond["cond"].update(concat_conditioning)
        return cond

    # simulates a latent sample
    @pytest.fixture()
    def sample(self):
        return torch.rand(2, 6, 32, 32).to(DEVICE)

    # simulates a timestep
    @pytest.fixture(
        params=[10.0, torch.randint(1000, (2,), dtype=torch.float).to(DEVICE), 3]
    )
    def timesteps(self, request):
        return request.param

    def test_unet2d_cond_wrapper(self, sample, timesteps, conditioning):
        # for concat
        in_channels = (
            sample.shape[1] + conditioning["cond"]["concat"].shape[1]
            if conditioning["cond"].get("concat", None) is not None
            else sample.shape[1]
        )

        # for vector
        class_embed_type = (
            "projection" if conditioning["cond"].get("vector") is not None else None
        )
        projection_class_embeddings_input_dim = (
            conditioning["cond"]["vector"].shape[1]
            if conditioning["cond"].get("vector") is not None
            else None
        )

        # for crossattn
        cross_attention_dim = (
            conditioning["cond"]["crossattn"].shape[2]
            if conditioning["cond"].get("crossattn") is not None
            else 1280
        )

        unet = DiffusersUNet2DCondWrapper(
            sample_size=sample.shape[2:],
            in_channels=in_channels,
            out_channels=3,
            class_embed_type=class_embed_type,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            cross_attention_dim=cross_attention_dim,
        ).to(DEVICE)
        output = unet(sample, timesteps, conditioning)
        assert output.shape == (
            sample.shape[0],
            3,
            sample.shape[2],
            sample.shape[3],
        )
