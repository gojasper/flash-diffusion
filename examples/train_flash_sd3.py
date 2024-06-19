import datetime
import os
from copy import deepcopy

import braceexpand
import torch.nn as nn
import yaml
from diffusers import (
    FlashFlowMatchEulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)
from peft import LoraConfig, get_peft_model
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from flash.data.datasets import DataModule, DataModuleConfig
from flash.data.filters import (
    FilterOnCondition,
    FilterOnConditionConfig,
    KeyFilter,
    KeyFilterConfig,
)
from flash.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    KeysFromJSONMapper,
    KeysFromJSONMapperConfig,
    MapperWrapper,
    RemoveKeysMapper,
    RemoveKeysMapperConfig,
    RescaleMapper,
    RescaleMapperConfig,
    SelectKeysMapper,
    SelectKeysMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from flash.models.flash_sd3 import FlashDiffusionSD3, FlashDiffusionSD3Config
from flash.models.transformers import DiffusersSD3Transformer2DWrapper
from flash.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from flash.trainer import TrainingConfig, TrainingPipeline
from flash.trainer.loggers import WandbSampleLogger


def main(args):
    # Load pretrained model as base
    if not args["USE_T5"]:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            text_encoder_3=None,
            tokenizer_3=None,
            revision="refs/pr/26",
            # torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            # torch_dtype=torch.float16,
            revision="refs/pr/26",
        )

    ### UNet ###
    # Get Architecture
    teacher_denoiser = DiffusersSD3Transformer2DWrapper(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=24,
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=192,
    )

    teacher_denoiser.load_state_dict(pipe.transformer.state_dict(), strict=True)

    del pipe.transformer
    del pipe.vae

    conditioner = None

    student_denoiser = deepcopy(teacher_denoiser)

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version="stabilityai/stable-diffusion-3-medium",
        revision="refs/pr/26",
        subfolder="vae",
        # torch_dtype=torch.float16,
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()

    if args["LORA"]:
        # LoRA config ##
        lora_config = LoraConfig(
            r=args["LORA_RANK"],
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "proj",
                "linear",
                "linear_1",
                "linear_2",
            ],
        )

        student_denoiser = get_peft_model(student_denoiser, lora_config)
        student_denoiser.print_trainable_parameters()

    teacher_denoiser.freeze()
    teacher_scheduler = eval(args["TEACHER_SCHEDULER"]).from_pretrained(
        "stabilityai/stable-diffusion-3-medium",
        revision="refs/pr/26",
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    sampling_scheduler = eval(args["SAMPLING_SCHEDULER"]).from_pretrained(
        "stabilityai/stable-diffusion-3-medium",
        revision="refs/pr/26",
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    teacher_sampling_scheduler = eval(
        args["TEACHER_SAMPLING_SCHEDULER"]
    ).from_pretrained(
        "stabilityai/stable-diffusion-3-medium",
        revision="refs/pr/26",
        subfolder="scheduler",
    )

    # discriminator
    discriminator_feature_dim = 64
    color_dim = 16

    discriminator = nn.Sequential(
        nn.Conv2d(color_dim, discriminator_feature_dim, 4, 2, 1, bias=False),
        nn.SiLU(True),
        nn.Conv2d(
            discriminator_feature_dim,
            discriminator_feature_dim * 2,
            4,
            2,
            1,
            bias=False,
        ),
        nn.GroupNorm(4, discriminator_feature_dim * 2),
        nn.SiLU(True),
        nn.Conv2d(
            discriminator_feature_dim * 2,
            discriminator_feature_dim * 4,
            4,
            2,
            1,
            bias=False,
        ),
        nn.GroupNorm(4, discriminator_feature_dim * 4),
        nn.SiLU(True),
        nn.Conv2d(
            discriminator_feature_dim * 4,
            discriminator_feature_dim * 8,
            4,
            2,
            1,
            bias=False,
        ),
        nn.GroupNorm(4, discriminator_feature_dim * 8),
        nn.SiLU(True),
        nn.Conv2d(discriminator_feature_dim * 8, 1, 4, 1, 0, bias=False),
        nn.Flatten(),
    )

    ## Diffusion Model ##
    # Get diffusion model
    config = FlashDiffusionSD3Config(
        K=args["K"],
        num_iterations_per_K=args["NUM_ITERATIONS_PER_K"],
        guidance_scale_min=args["GUIDANCE_MIN"],
        guidance_scale_max=args["GUIDANCE_MAX"],
        timestep_distribution=args["TIMESTEP_DISTRIBUTION"],
        mixture_num_components=args["MIXTURE_NUM_COMPONENTS"],
        mixture_var=args["MIXTURE_VAR"],
        use_dmd_loss=args["USE_DMD_LOSS"],
        dmd_loss_scale=args["DMD_LOSS_SCALE"],
        distill_loss_type=args["DISTILL_LOSS_TYPE"],
        distill_loss_scale=args["DISTILL_LOSS_SCALE"],
        adversarial_loss_scale=args["ADVERSARIAL_LOSS_SCALE"],
        gan_loss_type=args["GAN_LOSS_TYPE"],
        mode_probs=args["MODE_PROBS"],
        use_teacher_as_real=args["USE_TEACHER_AS_REAL"],
    )
    model = FlashDiffusionSD3(
        config,
        student_denoiser=student_denoiser,
        teacher_denoiser=teacher_denoiser,
        teacher_noise_scheduler=teacher_scheduler,
        sampling_noise_scheduler=sampling_scheduler,
        teacher_sampling_noise_scheduler=teacher_sampling_scheduler,
        vae=vae,
        conditioner=conditioner,
        discriminator=discriminator,
        pipeline=pipe,
        cpu_offload=args["CPU_OFFLOAD"],
    )

    ##################### DATA #####################

    # Define filters and mappers
    filters_mappers = [
        KeyFilter(KeyFilterConfig(keys=["jpg", "json"])),
        SelectKeysMapper(SelectKeysMapperConfig(keys=["jpg", "json"])),
        MapperWrapper(
            [
                KeysFromJSONMapper(
                    KeysFromJSONMapperConfig(
                        key="json",
                        keys_to_extract=[
                            "caption",
                            "aesthetic_score",
                        ],
                        remove_original=False,
                        strict=False,
                    )
                ),
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            "jpg": "image",
                            "caption": "text",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["CenterCrop", "ToTensor"],
                        transforms_kwargs=[
                            {"size": (1024, 1024)},
                            {},
                        ],
                    )
                ),
                RemoveKeysMapper(RemoveKeysMapperConfig(keys=["json"])),
                RescaleMapper(RescaleMapperConfig(key="image")),
            ]
        ),
        FilterOnCondition(
            FilterOnConditionConfig(
                condition_key="aesthetic_score",
                condition_fn=lambda x: x >= 6.0,
            )
        ),
    ]

    shards_path_or_urls = args["SHARDS_PATH_OR_URLS"]

    # unbrace urls
    shards_path_or_urls_unbraced = []
    for shards_path_or_url in shards_path_or_urls:
        shards_path_or_urls_unbraced.extend(braceexpand.braceexpand(shards_path_or_url))

    data_module = DataModule(
        train_config=DataModuleConfig(
            shards_path_or_urls=shards_path_or_urls_unbraced,
            decoder="pil",
            per_worker_batch_size=args["BATCH_SIZE"],
            shuffle_after_filter_mappers_buffer_size=20,
            shuffle_before_filter_mappers_buffer_size=20,
            shuffle_before_split_by_node_buffer_size=20,
            shuffle_before_split_by_workers_buffer_size=20,
            num_workers=4,
        ),
        train_filters_mappers=filters_mappers,
    )

    ##################### TRAIN #####################
    discriminator_trainable_params = [
        "discriminator.",
    ]

    # Training Config
    training_config = TrainingConfig(
        optimizers_name=["AdamW", "AdamW"],
        learning_rates=[args["LR"], args["LR_DISCRIMINATOR"]],
        log_keys=["image", "text"],
        trainable_params=[["student_denoiser"], discriminator_trainable_params],
        log_samples_model_kwargs={
            "max_samples": 8,
            "num_steps": args["NUM_STEPS"],
            "teacher_guidance_scale": args["TEACHER_SAMPLING_GUIDANCE_SCALE"],
            "log_teacher_samples": args["LOG_TEACHER_SAMPLES"],
            "conditioner_inputs": {
                "text": args["VALIDATION_PROMPTS"],
            },
        },
    )
    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-"
        + args["EXP_NAME"]
        + "-"
        + f"{os.environ['SLURM_JOB_ID']}"
    )
    dir_path = f"logs/{training_signature}"
    ckpt_path = f"{dir_path}/checkpoints"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    run_name = training_signature

    trainer = Trainer(
        accelerator="gpu",
        devices=int(os.environ["SLURM_NPROCS"]) // int(os.environ["SLURM_NNODES"]),
        num_nodes=int(os.environ["SLURM_NNODES"]),
        strategy="ddp_find_unused_parameters_true",
        default_root_dir=dir_path,
        max_epochs=args["MAX_EPOCHS"],
        logger=loggers.WandbLogger(
            project="flash-diffusion",
            offline=False,
            save_dir=dir_path,
            name=run_name,
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=args["LOG_EVERY_N_BATCHES"]),
            ModelCheckpoint(
                dirpath=ckpt_path,
                filename="{step}",
                every_n_train_steps=args["CKPT_EVERY_N_STEPS"],
                save_top_k=-1,  # to save all the models
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        check_val_every_n_epoch=100000000,
    )

    trainer.fit(pipeline, data_module)


if __name__ == "__main__":
    config_path = "configs/flash_sd3.yaml"
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
        print(args)
        main(args)
