import datetime
import os
from copy import deepcopy

import braceexpand
import torch
import torch.nn as nn
import yaml
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    LCMScheduler,
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
    SetValueConfig,
    SetValueMapper,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from flash.models.embedders import (
    ConditionerWrapper,
    T5TextEmbedder,
    T5TextEmbedderConfig,
    TimestepsEmbedder,
    TimestepsEmbedderConfig,
)
from flash.models.flash import FlashDiffusion, FlashDiffusionConfig
from flash.models.transformers import DiffusersTransformer2DWrapper
from flash.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from flash.trainer import TrainingConfig, TrainingPipeline
from flash.trainer.loggers import WandbSampleLogger


def main(args):
    # Load pretrained model as base
    pipe = DiffusionPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
    )

    ### UNet ###
    # Get Architecture
    teacher_denoiser = DiffusersTransformer2DWrapper(
        sample_size=128,
        num_layers=28,
        attention_head_dim=72,
        in_channels=4,
        out_channels=8,
        patch_size=2,
        attention_bias=True,
        num_attention_heads=16,
        cross_attention_dim=1152,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        caption_channels=4096,
        projection_class_embeddings_input_dim=256,
        time_embed_dim=1152,  # attention_head_dim * num_attention_heads
        timesteps_embedding_num_channels=256,
        use_concat_vector_conditioning=True,
        num_vector_conditionings=3,
    )

    teacher_denoiser.load_state_dict(pipe.transformer.state_dict(), strict=False)

    teacher_denoiser.adaln_single.timestep_embedder.linear_1.weight.data = (
        pipe.transformer.state_dict()[
            "adaln_single.emb.timestep_embedder.linear_1.weight"
        ]
    )
    teacher_denoiser.adaln_single.timestep_embedder.linear_1.bias.data = (
        pipe.transformer.state_dict()[
            "adaln_single.emb.timestep_embedder.linear_1.bias"
        ]
    )
    teacher_denoiser.adaln_single.timestep_embedder.linear_2.weight.data = (
        pipe.transformer.state_dict()[
            "adaln_single.emb.timestep_embedder.linear_2.weight"
        ]
    )
    teacher_denoiser.adaln_single.timestep_embedder.linear_2.bias.data = (
        pipe.transformer.state_dict()[
            "adaln_single.emb.timestep_embedder.linear_2.bias"
        ]
    )

    teacher_denoiser.adaln_single.add_embedding[
        0
    ].linear_1.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_1.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        0
    ].linear_1.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_1.bias"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        0
    ].linear_2.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_2.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        0
    ].linear_2.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_2.bias"
    ]

    teacher_denoiser.adaln_single.add_embedding[
        1
    ].linear_1.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_1.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        1
    ].linear_1.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_1.bias"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        1
    ].linear_2.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_2.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        1
    ].linear_2.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.resolution_embedder.linear_2.bias"
    ]

    teacher_denoiser.adaln_single.add_embedding[
        2
    ].linear_1.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.aspect_ratio_embedder.linear_1.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        2
    ].linear_1.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.aspect_ratio_embedder.linear_1.bias"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        2
    ].linear_2.weight.data = pipe.transformer.state_dict()[
        "adaln_single.emb.aspect_ratio_embedder.linear_2.weight"
    ]
    teacher_denoiser.adaln_single.add_embedding[
        2
    ].linear_2.bias.data = pipe.transformer.state_dict()[
        "adaln_single.emb.aspect_ratio_embedder.linear_2.bias"
    ]

    student_denoiser = deepcopy(teacher_denoiser)

    ## CONDITIONERS ##
    # Get text encoders
    text_embedder_config = T5TextEmbedderConfig(
        version="PixArt-alpha/PixArt-XL-2-1024-MS",
        text_embedder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        tokenizer_max_length=120,
        returns_attention_mask=True,
        unconditional_conditioning_rate=0.0,
    )
    text_embedder = T5TextEmbedder(text_embedder_config)

    # Freeze text encoders
    text_embedder.freeze()

    # Crop cooords and target/original shapes
    time_embedder_config_1 = TimestepsEmbedderConfig(
        input_dim=256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        input_key="resolution_height",
    )
    time_embedder_1 = TimestepsEmbedder(time_embedder_config_1)

    time_embedder_config_2 = TimestepsEmbedderConfig(
        input_dim=256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        input_key="resolution_width",
    )
    time_embedder_2 = TimestepsEmbedder(time_embedder_config_2)

    time_embedder_config_3 = TimestepsEmbedderConfig(
        input_dim=256,
        flip_sin_to_cos=True,
        downscale_freq_shift=0,
        input_key="aspect_ratio",
    )
    time_embedder_3 = TimestepsEmbedder(time_embedder_config_3)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=[
            text_embedder,
            time_embedder_1,
            time_embedder_2,
            time_embedder_3,
        ],
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version="PixArt-alpha/PixArt-XL-2-1024-MS",
        subfolder="vae",
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()

    if args["LORA"]:
        # LoRA config ##
        lora_config = LoraConfig(
            r=64,
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
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    sampling_scheduler = eval(args["SAMPLING_SCHEDULER"]).from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
    teacher_sampling_scheduler = eval(
        args["TEACHER_SAMPLING_SCHEDULER"]
    ).from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        subfolder="scheduler",
    )

    # Discriminator
    discriminator_feature_dim = 64
    color_dim = 4

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
        nn.Conv2d(
            discriminator_feature_dim * 8,
            discriminator_feature_dim * 16,
            4,
            2,
            1,
            bias=False,
        ),
        nn.GroupNorm(4, discriminator_feature_dim * 16),
        nn.SiLU(True),
        nn.Conv2d(discriminator_feature_dim * 16, 1, 4, 1, 0, bias=False),
        nn.Flatten(),
    )

    ## Diffusion Model ##
    # Get diffusion model
    config = FlashDiffusionConfig(
        ucg_keys=args["UCG_KEYS"],
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
        use_empty_prompt=args["USE_EMPTY_PROMPT"],
    )
    model = FlashDiffusion(
        config,
        student_denoiser=student_denoiser,
        teacher_denoiser=teacher_denoiser,
        teacher_noise_scheduler=teacher_scheduler,
        sampling_noise_scheduler=sampling_scheduler,
        teacher_sampling_noise_scheduler=teacher_sampling_scheduler,
        vae=vae,
        conditioner=conditioner,
        discriminator=discriminator,
    )

    del pipe

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
                SetValueMapper(
                    SetValueConfig(
                        key="resolution_height", value=torch.tensor([1024.0])
                    )
                ),
                SetValueMapper(
                    SetValueConfig(key="resolution_width", value=torch.tensor([1024.0]))
                ),
                SetValueMapper(
                    SetValueConfig(key="aspect_ratio", value=torch.tensor([1.0]))
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
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + args["EXP_NAME"]
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
    config_path = "configs/flash_pixart.yaml"
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)
        main(args)
