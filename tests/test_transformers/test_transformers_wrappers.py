import pytest
import torch

from flash.models.transformers import DiffusersTransformer2DWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestDiffusersTransformer2DWrapper:
    # simulates class conditioning
    @pytest.fixture(params=[None, torch.randint(256, (2, 12)).to(DEVICE).float()])
    def vector_conditioning(self, request):
        if request.param is not None:
            return {"vector": request.param}
        return None

    # simulates crossattn conditioning
    @pytest.fixture(params=[None, torch.randn(2, 12, 123).to(DEVICE)])
    def crossattn_conditioning(self, request):
        if request.param is not None:
            return {"crossattn": request.param}
        return None

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
        cond = dict(cond={})
        if crossattn_conditioning is not None:
            cond["cond"].update(crossattn_conditioning)
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
    @pytest.fixture(params=[torch.randint(1000, (2,), dtype=torch.float).to(DEVICE)])
    def timesteps(self, request):
        return request.param

    def test_transformer2d_wrapper(self, sample, timesteps, conditioning):
        # for concat
        in_channels = (
            sample.shape[1] + conditioning["cond"]["concat"].shape[1]
            if conditioning["cond"].get("concat", None) is not None
            else sample.shape[1]
        )

        # for vector
        projection_class_embeddings_input_dim = (
            conditioning["cond"]["vector"].shape[1]
            if conditioning["cond"].get("vector") is not None
            else None
        )
        # for crossattn
        cross_attention_dim = (
            conditioning["cond"]["crossattn"].shape[2]
            if conditioning["cond"].get("crossattn") is not None
            else None
        )
        transformer = DiffusersTransformer2DWrapper(
            in_channels=in_channels,
            attention_head_dim=64,
            num_attention_heads=8,
            num_layers=3,
            out_channels=3,
            cross_attention_dim=cross_attention_dim,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            patch_size=2,
            sample_size=sample.shape[-1],
            time_embed_dim=512,
            norm_type="ada_norm_single",  # pixart-alpha style,
            double_self_attention=False if cross_attention_dim is not None else True,
        ).to(DEVICE)
        output = transformer(sample, timesteps, conditioning)
        assert output.shape == (
            sample.shape[0],
            3,
            sample.shape[2],
            sample.shape[3],
        )
