from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from flash.models.embedders import ClipEmbedder, ClipEmbedderConfig, ConditionerWrapper
from flash.models.flash import FlashDiffusion, FlashDiffusionConfig
from flash.models.unets import DiffusersUNet2DCondWrapper
from flash.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestTurbo:

    @pytest.fixture()
    def embedder1(self):
        return ClipEmbedder(
            ClipEmbedderConfig(
                version="stabilityai/stable-diffusion-xl-base-1.0",
                text_embedder_subfolder="text_encoder",
                tokenizer_subfolder="tokenizer",
                input_key="text",
                unconditional_conditioning_rate=0.0,
            )
        )

    @pytest.fixture()
    def embedder2(self):
        return ClipEmbedder(
            ClipEmbedderConfig(
                version="stabilityai/stable-diffusion-xl-base-1.0",
                text_embedder_subfolder="text_encoder_2",
                tokenizer_subfolder="tokenizer_2",
                input_key="text",
                always_return_pooled=True,
                unconditional_conditioning_rate=0.0,
            )
        )

    @pytest.fixture()
    def teacher_unet(self):
        return DiffusersUNet2DCondWrapper(
            in_channels=4,  # VAE channels
            out_channels=4,  # VAE channels
            cross_attention_dim=768 + 1280,  # 2 text encoders
            projection_class_embeddings_input_dim=1280,  # 1 pooled text encoder
            class_embed_type="projection",
        )

    @pytest.fixture()
    def student_unet(self):
        return DiffusersUNet2DCondWrapper(
            in_channels=4,  # VAE channels
            out_channels=4,  # VAE channels
            cross_attention_dim=768 + 1280,  # 2 text encoders
            projection_class_embeddings_input_dim=1280,  # 1 pooled text encoder
            class_embed_type="projection",
        )

    @pytest.fixture()
    def conditioner_wrapper(self, embedder1, embedder2):
        conditioner = ConditionerWrapper(conditioners=[embedder1, embedder2]).to(DEVICE)
        return conditioner

    @pytest.fixture()
    def model_config(self):
        return FlashDiffusionConfig(
            K=[32],
            num_iterations_per_K=[5000],
            guidance_scale_min=3.0,
            guidance_scale_max=7.0,
            distill_loss_type="lpips",
            ucg_keys=["text"],
            timestep_distribution="gaussian",
            allow_full_noise=False,
            switch_teacher=False,
            mixture_num_components=8,
            mixture_var=0.5,
            adapter_conditioning_scale=1.0,
            adapter_input_key=None,
            controlnet_input_key=None,
            adversarial_loss_scale=1.0,
        )

    @pytest.fixture()
    def vae(self):
        return AutoencoderKLDiffusers(AutoencoderKLDiffusersConfig())

    @pytest.fixture()
    def teacher_noise_scheduler(self):
        return DDPMScheduler()

    @pytest.fixture()
    def student_noise_scheduler(self):
        return DDPMScheduler()

    @pytest.fixture()
    def discriminator(self):
        discriminator_feature_dim = 64

        color_dim = 1280

        discriminator = nn.Sequential(
            nn.Conv2d(color_dim, discriminator_feature_dim, 4, 2, 1, bias=False),
            nn.SiLU(True),
            nn.Conv2d(discriminator_feature_dim, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
        )

        return discriminator

    @pytest.fixture()
    def model_input(self):
        return {
            "image": torch.randn(2, 3, 512, 512).to(DEVICE),
            "text": ["Test clip", "Test clip with 2 texts"],
        }

    @pytest.fixture()
    def turbo_model(
        self,
        model_config,
        teacher_unet,
        student_unet,
        teacher_noise_scheduler,
        student_noise_scheduler,
        vae,
        conditioner_wrapper,
        discriminator,
    ):
        return FlashDiffusion(
            config=model_config,
            student_denoiser=student_unet,
            teacher_denoiser=teacher_unet,
            teacher_noise_scheduler=student_noise_scheduler,
            sampling_noise_scheduler=teacher_noise_scheduler,
            teacher_sampling_noise_scheduler=teacher_noise_scheduler,
            vae=vae,
            conditioner=conditioner_wrapper,
            discriminator=discriminator,
        ).to(DEVICE)

    @torch.no_grad()
    def test_model_forward(self, turbo_model, model_input):
        model_output = turbo_model(model_input, device=DEVICE, step=0)
        assert model_output["loss"][0] > 0.0
        assert model_output["loss"][1] == 0.0
        model_output = turbo_model(model_input, device=DEVICE, step=1)
        assert model_output["loss"][0] > 0.0
        assert model_output["loss"][1] > 0.0

    def test_optimizers(self, turbo_model, model_input):
        optimizer = torch.optim.Adam(turbo_model.student_denoiser.parameters(), lr=1e-4)
        optimizer_discriminator = torch.optim.Adam(
            turbo_model.discriminator.parameters(), lr=1e-4
        )
        turbo_model.train()
        turbo_model_init = deepcopy(turbo_model)
        optimizer.zero_grad()
        loss = turbo_model(model_input, device=DEVICE, step=0)["loss"][0]
        loss.backward()
        optimizer.step()
        optimizer_discriminator.zero_grad()
        loss = turbo_model(model_input, device=DEVICE, step=1)["loss"][1]
        loss.backward()
        optimizer_discriminator.step()
        assert not torch.equal(
            torch.cat([p.flatten() for p in turbo_model.student_denoiser.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.student_denoiser.parameters()]
            ),
        )
        assert torch.equal(
            torch.cat([p.flatten() for p in turbo_model.teacher_denoiser.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.teacher_denoiser.parameters()]
            ),
        )
        assert not torch.equal(
            torch.cat([p.flatten() for p in turbo_model.discriminator.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.discriminator.parameters()]
            ),
        )

    def test_optimizers_no_reg(self, turbo_model, model_input):
        turbo_model.distill_loss_scale = [0.0]
        optimizer = torch.optim.Adam(turbo_model.student_denoiser.parameters(), lr=1e-4)
        optimizer_discriminator = torch.optim.Adam(
            turbo_model.discriminator.parameters(), lr=1e-4
        )
        turbo_model.train()
        turbo_model_init = deepcopy(turbo_model)
        optimizer.zero_grad()
        loss = turbo_model(model_input, device=DEVICE, step=0)["loss"][0]
        loss.backward()
        optimizer.step()
        optimizer_discriminator.zero_grad()
        loss = turbo_model(model_input, device=DEVICE, step=1)["loss"][1]
        loss.backward()
        optimizer_discriminator.step()
        assert not torch.equal(
            torch.cat([p.flatten() for p in turbo_model.student_denoiser.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.student_denoiser.parameters()]
            ),
        )
        assert torch.equal(
            torch.cat([p.flatten() for p in turbo_model.teacher_denoiser.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.teacher_denoiser.parameters()]
            ),
        )
        assert not torch.equal(
            torch.cat([p.flatten() for p in turbo_model.discriminator.parameters()]),
            torch.cat(
                [p.flatten() for p in turbo_model_init.discriminator.parameters()]
            ),
        )
