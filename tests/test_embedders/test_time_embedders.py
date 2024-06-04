import pytest
import torch

from flash.models.embedders import TimestepsEmbedder, TimestepsEmbedderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestTimeStepsEmbedder:
    @pytest.fixture()
    def config(self):
        return TimestepsEmbedderConfig(
            num_channelsx=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            input_key="input_size",
        )

    @pytest.fixture()
    def conditioner_input(self):
        return {
            "input_size": torch.tensor([[1024, 1024], [256, 256]]).to(DEVICE),
            "image": torch.randn(2, 3, 32, 32).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    def test_instanciate(self, config, conditioner_input):
        embedder = TimestepsEmbedder(config)
        output = embedder(conditioner_input)
        assert len(output) == 1
        assert output["vector"].shape == (
            conditioner_input["input_size"].shape[0],
            conditioner_input["input_size"].shape[1] * config.num_channels,
        )
