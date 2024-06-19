import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.schedulers import DDPMScheduler, LCMScheduler
from tqdm import tqdm

from ..base.base_model import BaseModel
from ..embedders import ConditionerWrapper
from ..transformers import DiffusersTransformer2DWrapper
from ..unets import DiffusersUNet2DCondWrapper, DiffusersUNet2DWrapper
from ..vae import AutoencoderKLDiffusers
from .flash_diffusion_config import FlashDiffusionSD3Config


def gaussian_mixture(locs, var, mode_probs=None):
    """
    Gaussian mixture distribution

    Args:

        locs (List[int]): The mean of each mode
        var (float): The variance of each mode
        mode_probs (Optional[List[float]]): The weight for each mode. If None, uniform weights are used.
    """
    if mode_probs is None:
        mode_probs = [1 / len(locs)] * len(locs)

    def _gaussian(x):
        prob = [
            mode_probs[i] * torch.exp(-torch.tensor([(x - loc) ** 2 / var]))
            for i, loc in enumerate(locs)
        ]
        return sum(prob)

    return _gaussian


class FlashDiffusionSD3(BaseModel):
    """
    Flash Diffusion model.

    A method to accelerate the sampling for conditional diffusion models.

    Args:

        config (FlashDiffusionSD3Config): The configuration class for the FlashDiffusion model.
        student_denoiser (Union[DiffusersUNet2DWrapper, DiffusersUNet2DCondWrapper, DiffusersTransformer2DWrapper]): The student denoiser model.
        teacher_denoiser (Union[DiffusersUNet2DWrapper, DiffusersUNet2DCondWrapper, DiffusersTransformer2DWrapper], optional): The teacher denoiser model. Defaults to None.
        teacher_noise_scheduler (DDPMScheduler): The teacher noise scheduler to generate synthetic sqmples during training. Defaults to None.
        teacher_sampling_noise_scheduler (DDPMScheduler): The teacher sampling noise scheduler to log samples generated with the teacher. Defaults to None.
        sampling_noise_scheduler (LCMScheduler): The sampling noise scheduler for the student. Typically LCMScheduler. Defaults to None.
        vae (AutoencoderKLDiffusers): The VAE model to encode the inputs. Defaults to None.
        conditioner (ConditionerWrapper): The conditioner model. Defaults to None.
        discriminator (torch.nn.Module): The discriminator model. Defaults to None.
        pipeline (DiffusionPipeline): The pipeline model. Defaults to None.
        cpu_offload (bool): Whether to offload the pipeline to CPU. Defaults to False.
    """

    def __init__(
        self,
        config: FlashDiffusionSD3Config,
        student_denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
            DiffusersTransformer2DWrapper,
        ],
        teacher_denoiser: Union[
            DiffusersUNet2DWrapper,
            DiffusersUNet2DCondWrapper,
            DiffusersTransformer2DWrapper,
        ] = None,
        teacher_noise_scheduler: DDPMScheduler = None,
        teacher_sampling_noise_scheduler: DDPMScheduler = None,
        sampling_noise_scheduler: LCMScheduler = None,
        vae: AutoencoderKLDiffusers = None,
        conditioner: ConditionerWrapper = None,
        discriminator: torch.nn.Module = None,
        pipeline: DiffusionPipeline = None,
        cpu_offload: bool = False,
    ):
        super().__init__(config)

        self.student_denoiser = student_denoiser
        self.teacher_denoiser = teacher_denoiser
        self.teacher_noise_scheduler = teacher_noise_scheduler
        self.teacher_sampling_noise_scheduler = teacher_sampling_noise_scheduler
        self.vae = vae
        self.conditioner = conditioner
        self.guidance_scale_min = config.guidance_scale_min
        self.guidance_scale_max = config.guidance_scale_max
        self.K = config.K
        self.num_iterations_per_K = config.num_iterations_per_K
        self.distill_loss_type = config.distill_loss_type
        self.timestep_distribution = config.timestep_distribution
        self.iter_steps = 0
        self.mixture_num_components = config.mixture_num_components
        self.mixture_var = config.mixture_var
        self.use_dmd_loss = config.use_dmd_loss
        self.dmd_loss_scale = config.dmd_loss_scale
        self.distill_loss_scale = config.distill_loss_scale
        self.discriminator = discriminator
        self.adversarial_loss_scale = config.adversarial_loss_scale
        self.gan_loss_type = config.gan_loss_type
        self.mode_probs = config.mode_probs
        self.use_teacher_as_real = config.use_teacher_as_real
        self.pipeline = pipeline
        self.cpu_offload = cpu_offload

        self.teacher_noise_scheduler_copy = deepcopy(self.teacher_noise_scheduler)

        self.disc_update_counter = 0

        if self.discriminator is None:
            logging.warning(
                "No discriminator provided. Adversarial loss will be ignored."
            )
            self.use_adversarial_loss = False

        else:
            self.use_adversarial_loss = True

        self.disc_backbone = self.teacher_denoiser

        if self.distill_loss_type == "lpips":
            self.lpips = lpips.LPIPS(net="vgg")

        self.sampling_noise_scheduler = sampling_noise_scheduler

        self.K_steps = np.cumsum(self.num_iterations_per_K)
        self.K_prev = self.K[0]

    def _encode_inputs(self, batch: Dict[str, Any]):
        """
        Encode the inputs using the VAE
        """
        with torch.no_grad():
            vae_inputs = batch[self.vae.config.input_key]
            return self.vae.encode(vae_inputs)

    def _get_timesteps(
        self, num_samples: int = 1, K: int = 1, K_step: int = 1, device="cpu"
    ):
        # Get the timesteps for the current K
        self.teacher_noise_scheduler.set_timesteps(K)

        if self.timestep_distribution == "uniform":
            prob = torch.ones(K) / K
        elif self.timestep_distribution == "gaussian":
            prob = [torch.exp(-torch.tensor([(i - K / 2) ** 2 / K])) for i in range(K)]
            prob = torch.tensor(prob) / torch.sum(torch.tensor(prob))
        elif self.timestep_distribution == "mixture":
            mixture_num_components = self.mixture_num_components[K_step]
            mode_probs = self.mode_probs[K_step]

            # Define targeted timesteps
            locs = [
                i * (K // mixture_num_components)
                for i in range(0, mixture_num_components)
            ]
            mixture_var = self.mixture_var[K_step]
            prob = [
                gaussian_mixture(
                    locs=locs,
                    var=mixture_var,
                    mode_probs=mode_probs,
                )(i)
                for i in range(K)
            ]
            prob = torch.tensor(prob) / torch.sum(torch.tensor(prob))

        start_idx = torch.multinomial(prob, 1)

        start_timestep = (
            self.teacher_noise_scheduler.timesteps[start_idx]
            .to(device)
            .repeat(num_samples)
        )

        return start_idx, start_timestep

    def forward(self, batch: Dict[str, Any], batch_idx=0, step=0, *args, **kwargs):

        self.iter_steps += 1
        if self.vae is not None:
            z = self._encode_inputs(batch)
        else:
            z = batch[self.input_key]

        self.pipeline.to(z.device)

        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipeline.encode_prompt(
                prompt=batch["text"],
                prompt_2=batch["text"],
                prompt_3=batch["text"],
                negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                negative_prompt_2="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                negative_prompt_3="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
                do_classifier_free_guidance=True,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                clip_skip=False,
                device=z.device,
            )

        if self.cpu_offload:
            self.pipeline.to("cpu")

        conditioning = {
            "cond": {"vector": pooled_prompt_embeds, "crossattn": prompt_embeds}
        }

        student_conditioning = conditioning

        unconditional_conditioning = {
            "cond": {
                "vector": negative_pooled_prompt_embeds,
                "crossattn": negative_prompt_embeds,
            }
        }

        # Get K for the current step
        if self.iter_steps > self.K_steps[-1]:
            K_step = len(self.K) - 1
        else:
            K_step = np.argmax(self.iter_steps < self.K_steps)
        K = self.K[K_step]
        guidance_min = self.guidance_scale_min[K_step]
        guidance_max = self.guidance_scale_max[K_step]
        if K != self.K_prev:
            self.K_prev = K
            if self.switch_teacher:
                print("Switching teacher")
                self.teacher_denoiser = deepcopy(self.student_denoiser)
                self.teacher_denoiser.freeze()

        # Create noisy samples
        noise = torch.randn_like(z)

        # Sample the timesteps
        start_idx, start_timestep = self._get_timesteps(
            num_samples=z.shape[0], K=K, K_step=K_step, device=z.device
        )

        # Add noise to sample
        sigmas = self.get_sigmas(
            self.teacher_noise_scheduler, start_timestep, device=z.device
        )

        if start_idx == 0:
            noisy_sample_init = noise
            if hasattr(self.teacher_noise_scheduler, "init_noise_sigma"):
                noisy_sample_init *= self.teacher_noise_scheduler.init_noise_sigma
            noisy_sample_init_student = noise

        else:
            noisy_sample_init = sigmas * noise + (1.0 - sigmas) * z
            noisy_sample_init_student = noisy_sample_init

        noisy_sample_init_ = noisy_sample_init_student

        noisy_sample = noisy_sample_init.clone().detach()

        guidance_scale = (
            torch.rand(1).to(z.device) * (guidance_max - guidance_min) + guidance_min
        )

        with torch.no_grad():
            for t in self.teacher_noise_scheduler.timesteps[start_idx:]:
                timestep = torch.tensor([t], device=z.device).repeat(z.shape[0])
                # print(timestep)

                noisy_sample_ = noisy_sample

                # Denoise sample
                cond_noise_pred = self.teacher_denoiser(
                    sample=noisy_sample_,
                    timestep=timestep,
                    conditioning=conditioning,
                    *args,
                    **kwargs,
                )

                uncond_noise_pred = self.teacher_denoiser(
                    sample=noisy_sample_,
                    timestep=timestep,
                    conditioning=unconditional_conditioning,
                    *args,
                    **kwargs,
                )

                # Make CFG
                noise_pred = (
                    guidance_scale * cond_noise_pred
                    + (1 - guidance_scale) * uncond_noise_pred
                )

                # Make one step on the reverse diffusion process
                noisy_sample = self.teacher_noise_scheduler.step(
                    noise_pred, t, noisy_sample, return_dict=False
                )[0]

        teacher_output = noisy_sample

        # Get student denoiser output
        student_noise_pred = self.student_denoiser(
            sample=noisy_sample_init_,
            timestep=start_timestep,
            conditioning=student_conditioning,
        )

        student_output = noisy_sample_init_ - student_noise_pred * sigmas

        loss = (
            self._distill_loss(student_output, teacher_output)
            * self.distill_loss_scale[K_step]
        )

        if self.use_dmd_loss:
            dmd_loss = self._dmd_loss(
                student_output,
                student_conditioning,
                conditioning,
                unconditional_conditioning,
                K,
                K_step,
            )
            loss += dmd_loss * self.dmd_loss_scale[K_step]

        if self.use_adversarial_loss:

            gan_loss = self._gan_loss(
                z,
                batch,
                student_output,
                teacher_output,
                conditioning,
                step=step,
            )
            loss += self.adversarial_loss_scale[K_step] * gan_loss[0]
            loss_disc = gan_loss[1]

            return {
                "loss": [loss, loss_disc],
                "teacher_output": teacher_output,
                "student_output": student_output,
                "noisy_sample": noisy_sample_init,
                "start_timestep": start_timestep[0].item(),
            }

        else:
            return {
                "loss": loss.mean(),
                "teacher_output": teacher_output,
                "student_output": student_output,
                "noisy_sample": noisy_sample_init,
                "start_timestep": start_timestep[0].item(),
            }

    def _distill_loss(self, student_output, teacher_output):
        """
        Compute the distillation loss
        """
        if self.distill_loss_type == "l2":
            return torch.mean(
                ((student_output - teacher_output) ** 2).reshape(
                    student_output.shape[0], -1
                ),
                1,
            ).mean()
        elif self.distill_loss_type == "l1":
            return torch.mean(
                torch.abs(student_output - teacher_output).reshape(
                    student_output.shape[0], -1
                ),
                1,
            ).mean()
        elif self.distill_loss_type == "lpips":
            # center crop patches of size 64x64
            crop_h = max((student_output.shape[2] - 64) // 2, 0)
            crop_w = max((student_output.shape[3] - 64) // 2, 0)

            student_output = student_output[
                :,
                :,
                crop_h : min(crop_h + 64, student_output.shape[2]),
                crop_w : min(crop_w + 64, student_output.shape[3]),
            ]
            teacher_output = teacher_output[
                :,
                :,
                crop_h : min(crop_h + 64, teacher_output.shape[2]),
                crop_w : min(crop_w + 64, teacher_output.shape[3]),
            ]

            decoded_student = self.vae.decode(student_output).clamp(-1, 1)
            decoded_teacher = self.vae.decode(teacher_output).clamp(-1, 1)
            return self.lpips(decoded_student, decoded_teacher).mean()
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

    def _dmd_loss(
        self,
        student_output,
        student_conditioning,
        conditioning,
        unconditional_conditioning,
        K,
        K_step,
    ):
        """
        Compute the DMD loss
        """

        # Sample noise
        noise = torch.randn_like(student_output)

        timestep = torch.randint(
            0,
            self.teacher_noise_scheduler.config.num_train_timesteps,
            (student_output.shape[0],),
            device="cpu",
        )
        timestep = self.teacher_noise_scheduler_copy.timesteps[timestep].to(
            student_output.device
        )

        # Create noisy sample
        sigmas = self.get_sigmas(
            self.teacher_noise_scheduler_copy, timestep, device=student_output.device
        )
        noisy_student = sigmas * noise + (1.0 - sigmas) * student_output

        with torch.no_grad():

            cond_real_noise_pred = self.teacher_denoiser(
                sample=noisy_student,
                timestep=timestep,
                conditioning=conditioning,
            )

            uncond_real_noise_pred = self.teacher_denoiser(
                sample=noisy_student,
                timestep=timestep,
                conditioning=unconditional_conditioning,
            )

            cond_fake_noise_pred = self.student_denoiser(
                sample=noisy_student,
                timestep=timestep,
                conditioning=student_conditioning,
            )

            guidance_scale = (
                torch.rand(1).to(student_output.device)
                * (self.guidance_scale_max[K_step] - self.guidance_scale_min[K_step])
                + self.guidance_scale_min[K_step]
            )

        real_noise_pred = (
            guidance_scale * cond_real_noise_pred
            + (1 - guidance_scale) * uncond_real_noise_pred
        )

        fake_noise_pred = cond_fake_noise_pred

        score_real = -real_noise_pred
        score_fake = -fake_noise_pred

        coeff = score_fake - score_real

        pred_x_0_student = real_noise_pred

        weight = (
            1.0
            / (
                (student_output - pred_x_0_student).abs().mean([1, 2, 3], keepdim=True)
                + 1e-5
            ).detach()
        )
        return F.mse_loss(
            student_output, (student_output - weight * coeff).detach(), reduction="mean"
        )

    def _gan_loss(
        self,
        z,
        batch,
        student_output,
        teacher_output,
        conditioning,
        step=0,
    ):
        """
        Compute the GAN loss
        """

        self.disc_update_counter += 1

        # Sample noise
        noise = torch.randn_like(student_output)

        if self.use_teacher_as_real:
            real = teacher_output

        else:
            real = z

        # Selected timesteps
        selected_timesteps = [
            float(self.teacher_noise_scheduler_copy.timesteps[-10]),
            float(self.teacher_noise_scheduler_copy.timesteps[-250]),
            float(self.teacher_noise_scheduler_copy.timesteps[-500]),
            float(self.teacher_noise_scheduler_copy.timesteps[-750]),
        ]
        prob = torch.tensor([0.25, 0.25, 0.25, 0.25])

        # Sample the timesteps
        idx = prob.multinomial(student_output.shape[0], replacement=True).to(
            student_output.device
        )
        timesteps = torch.tensor(selected_timesteps, device=student_output.device)[idx]

        # Create noisy sample
        sigmas = self.get_sigmas(
            self.teacher_noise_scheduler_copy, timesteps, device=student_output.device
        )
        noisy_fake = sigmas * noise + (1.0 - sigmas) * student_output
        noisy_real = sigmas * noise + (1.0 - sigmas) * real

        # Concatenate noisy samples
        noisy_sample = torch.cat([noisy_fake, noisy_real], dim=0)

        # Concatenate conditionings
        if conditioning is not None:
            conditioning = {
                "cond": {
                    k: torch.cat([v, v], dim=0) for k, v in conditioning["cond"].items()
                }
            }

        # Concatenate timesteps
        timestep = torch.cat([timesteps, timesteps], dim=0)

        # Predict noise level using denoiser
        denoised_sample = self.disc_backbone(
            sample=noisy_sample,
            timestep=timestep,
            conditioning=conditioning,
            return_post_mid_blocks=True,
        )

        denoised_sample_fake, denoised_sample_real = denoised_sample.chunk(2, dim=0)

        if self.gan_loss_type == "wgan":
            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            if step % 2 == 0:
                loss_G = -self.discriminator(denoised_sample_fake).mean()
                loss_D = 0
            else:
                loss_D = (
                    -self.discriminator(denoised_sample_real).mean()
                    + self.discriminator(denoised_sample_fake.detach()).mean()
                )
                loss_G = 0

        elif self.gan_loss_type == "lsgan":
            valid = torch.ones(student_output.size(0), 1, device=noise.device)
            fake = torch.zeros(noise.size(0), 1, device=noise.device)
            if step % 2 == 0:
                loss_G = F.mse_loss(
                    torch.sigmoid(self.discriminator(denoised_sample_fake)), valid
                )
                loss_D = 0
            else:
                loss_D = 0.5 * (
                    F.mse_loss(
                        torch.sigmoid(self.discriminator(denoised_sample_real)), valid
                    )
                    + F.mse_loss(
                        torch.sigmoid(
                            self.discriminator(denoised_sample_fake.detach())
                        ),
                        fake,
                    )
                )
                loss_G = 0
        elif self.gan_loss_type == "hinge":
            if step % 2 == 0:
                loss_G = -self.discriminator(denoised_sample_fake).mean()
                loss_D = 0
            else:
                loss_D = (
                    F.relu(1.0 - self.discriminator(denoised_sample_real)).mean()
                    + F.relu(
                        1.0 + self.discriminator(denoised_sample_fake.detach())
                    ).mean()
                )
                loss_G = 0

        elif self.gan_loss_type == "non-saturating":
            if step % 2 == 0:
                loss_G = -torch.mean(
                    torch.log(
                        torch.sigmoid(self.discriminator(denoised_sample_fake)) + 1e-8
                    )
                )
                loss_D = 0

            else:
                loss_D = -torch.mean(
                    torch.log(
                        torch.sigmoid(self.discriminator(denoised_sample_real)) + 1e-8
                    )
                    + torch.log(
                        1
                        - torch.sigmoid(
                            self.discriminator(denoised_sample_fake.detach())
                        )
                        + 1e-8
                    )
                )
                loss_G = 0
        else:
            if step % 2 == 0:
                valid = torch.ones(student_output.size(0), 1, device=noise.device)
                loss_G = F.binary_cross_entropy_with_logits(
                    self.discriminator(denoised_sample_fake), valid
                )
                loss_D = 0

            else:
                valid = torch.ones(student_output.size(0), 1, device=noise.device)
                real = F.binary_cross_entropy_with_logits(
                    self.discriminator(denoised_sample_real), valid
                )
                fake = torch.zeros(noise.size(0), 1, device=noise.device)
                fake = F.binary_cross_entropy_with_logits(
                    self.discriminator(denoised_sample_fake.detach()), fake
                )
                loss_D = real + fake
                loss_G = 0

        return [
            loss_G,
            loss_D,
        ]

    def _timestep_sampling(
        self, n_samples: int = 1, device="cpu", timestep_sampling="uniform"
    ) -> torch.Tensor:
        if timestep_sampling == "uniform":
            idx = self.prob.multinomial(n_samples, replacement=True).to(device)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

        elif timestep_sampling == "teacher":
            return torch.randint(
                0,
                self.teacher_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device=device,
            )

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 20,
        guidance_scale: float = 1.0,
        teacher_guidance_scale: float = 5.0,
        conditioner_inputs: Dict[str, Any] = None,
        uncond_conditioner_inputs: Dict[str, Any] = None,
        max_samples: int = None,
        verbose: bool = False,
        log_teacher_samples: bool = False,
    ):
        """
        Sample from the model
        Args:
            z (torch.Tensor): Noisy latent vector
            num_steps: (int): Number of steps to sample
            guidance_scale (float): Guidance scale for classiffier-free guidance. If 1, no guidance. Default: 1.0
            conditioner_inputs (Dict[str, Any]): inputs to the conditioners
            uncond_conditioner_inputs (Dict[str, Any]): inputs to the conditioner for CFG e.g. negative prompts.
            max_samples (Optional[int]): Maximum number of samples to generate. Default: None, all samples are generated
            verbose (bool): Whether to print progress bar. Default: True
        """

        self.teacher_noise_scheduler.set_timesteps(num_steps)

        self.sampling_noise_scheduler.set_timesteps(num_steps)
        print(self.sampling_noise_scheduler.timesteps)
        print(self.sampling_noise_scheduler.sigmas)

        sample = z

        self.pipeline.to(z.device)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=conditioner_inputs["text"],
            prompt_2=conditioner_inputs["text"],
            prompt_3=conditioner_inputs["text"],
            negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
            negative_prompt_2="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
            negative_prompt_3="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            clip_skip=False,
            device=z.device,
        )

        conditioning = {
            "cond": {"vector": pooled_prompt_embeds, "crossattn": prompt_embeds}
        }

        unconditional_conditioning = {
            "cond": {
                "vector": negative_pooled_prompt_embeds,
                "crossattn": negative_prompt_embeds,
            }
        }

        if max_samples is not None:
            sample = sample[:max_samples]

            if conditioning:
                conditioning["cond"] = {
                    k: v[:max_samples] for k, v in conditioning["cond"].items()
                }
                unconditional_conditioning["cond"] = {
                    k: v[:max_samples]
                    for k, v in unconditional_conditioning["cond"].items()
                }

        sample_init = sample
        if hasattr(self.sampling_noise_scheduler, "init_noise_sigma"):
            sample = sample * self.sampling_noise_scheduler.init_noise_sigma
        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):

            denoiser_input = sample

            # Predict noise level using denoiser using conditionings
            cond_noise_pred = self.student_denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=conditioning,
            )

            # Predict noise level using denoiser using unconditional conditionings
            uncond_noise_pred = self.student_denoiser(
                sample=denoiser_input,
                timestep=t.to(z.device).repeat(denoiser_input.shape[0]),
                conditioning=unconditional_conditioning,
            )

            # Make CFG
            noise_pred = (
                guidance_scale * cond_noise_pred
                + (1 - guidance_scale) * uncond_noise_pred
            )

            sample = self.sampling_noise_scheduler.step(
                noise_pred, t, sample, return_dict=False
            )[0]

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)
        else:
            decoded_sample = sample

        decoded_sample_ref = None

        if log_teacher_samples:
            self.teacher_sampling_noise_scheduler.set_timesteps(num_steps)

            if hasattr(self.teacher_sampling_noise_scheduler, "init_noise_sigma"):
                sample_ref = (
                    sample_init * self.teacher_sampling_noise_scheduler.init_noise_sigma
                )

            else:
                sample_ref = sample_init

            for i, t in tqdm(
                enumerate(self.teacher_sampling_noise_scheduler.timesteps),
                disable=not verbose,
            ):

                denoiser_input_ref = sample_ref

                cond_noise_pred_ref = self.teacher_denoiser(
                    sample=denoiser_input_ref,
                    timestep=t.to(z.device).repeat(denoiser_input_ref.shape[0]),
                    conditioning=conditioning,
                )
                uncond_noise_pred_ref = self.teacher_denoiser(
                    sample=denoiser_input_ref,
                    timestep=t.to(z.device).repeat(denoiser_input_ref.shape[0]),
                    conditioning=unconditional_conditioning,
                )

                noise_pred_ref = (
                    teacher_guidance_scale * cond_noise_pred_ref
                    + (1 - teacher_guidance_scale) * uncond_noise_pred_ref
                )
                sample_ref = self.teacher_sampling_noise_scheduler.step(
                    noise_pred_ref, t, sample_ref, return_dict=False
                )[0]

            if self.vae is not None:
                decoded_sample_ref = self.vae.decode(sample_ref)
            else:
                decoded_sample_ref = sample_ref

        return decoded_sample, decoded_sample_ref

    def log_samples(
        self,
        batch: Dict[str, Any],
        input_shape: Tuple[int, int, int] = None,
        guidance_scale: float = 1.0,
        teacher_guidance_scale: float = 5.0,
        max_samples: int = 8,
        num_steps: Union[int, List[int]] = 20,
        device="cpu",
        log_teacher_samples=False,
        conditioner_inputs: Dict = None,
        conditioner_uncond_inputs: Dict = None,
    ):

        if isinstance(num_steps, int):
            num_steps = [num_steps]

        logs = {}

        N = max_samples

        if batch is not None:
            max_conditioning_samples = min([len(batch[key]) for key in batch])
            N = min(N, max_conditioning_samples)

        if conditioner_inputs is not None:
            max_conditioning_samples = min(
                [len(conditioner_inputs[key]) for key in conditioner_inputs]
            )
            conditioner_inputs_ = {
                k: v.to(device)
                for k, v in conditioner_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            conditioner_inputs.update(conditioner_inputs_)
            batch.update(conditioner_inputs)
            N = min(N, max_conditioning_samples)

        if conditioner_uncond_inputs is not None:
            max_conditioning_samples = min(
                [
                    len(conditioner_uncond_inputs[key])
                    for key in conditioner_uncond_inputs
                ]
            )
            conditioner_uncond_inputs_ = {
                k: v.to(device)
                for k, v in conditioner_uncond_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            conditioner_uncond_inputs.update(conditioner_uncond_inputs_)
            batch_uncond = deepcopy(batch)
            batch_uncond.update(conditioner_uncond_inputs)
            N = min(N, max_conditioning_samples)
        else:
            batch_uncond = None

        # infer input shape based on VAE configuration if not passed
        if input_shape is None:
            if self.vae is not None:
                # get input pixel size of the vae
                input_shape = batch[self.vae.config.input_key].shape[2:]
                # rescale to latent size
                input_shape = (
                    self.vae.latent_channels,
                    input_shape[0] // self.vae.downsampling_factor,
                    input_shape[1] // self.vae.downsampling_factor,
                )
            else:
                raise ValueError(
                    "input_shape must be passed when no VAE is used in the model"
                )

        for num_step in num_steps:
            # Log samples
            z = torch.randn(N, *input_shape).to(device)

            logging.debug(
                f"Sampling {N} samples: steps={num_step}, guidance_scale={guidance_scale}"
            )
            samples, samples_ref = self.sample(
                z,
                num_steps=num_step,
                conditioner_inputs=batch,
                uncond_conditioner_inputs=batch_uncond,
                guidance_scale=guidance_scale,
                teacher_guidance_scale=teacher_guidance_scale,
                max_samples=N,
                log_teacher_samples=log_teacher_samples,
            )

            logs[
                f"samples_{num_step}_steps/{self.sampling_noise_scheduler.__class__.__name__}_{guidance_scale}_cfg/student"
            ] = samples

            if samples_ref is not None:
                logs[
                    f"samples_{num_step}_steps/{self.teacher_sampling_noise_scheduler.__class__.__name__}_{teacher_guidance_scale}_cfg/teacher"
                ] = samples_ref

        return logs

    def get_sigmas(
        self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
