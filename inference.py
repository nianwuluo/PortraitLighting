import argparse
import copy
import math
import random
from typing import Any
import pdb
import os
import cv2
import time
from PIL import Image, ImageOps

import torch
from accelerate import Accelerator
from library.device_utils import clean_memory_on_device
from safetensors.torch import load_file
from networks import lora_flux
from library import flux_models, flux_train_utils_relight as flux_train_utils, flux_utils, sd3_train_utils, \
    strategy_base, strategy_flux, train_util
import train_network
from library.utils import setup_logging
from diffusers.utils import load_image
import numpy as np
import torchvision.transforms as transforms

setup_logging()
import logging

logger = logging.getLogger(__name__)

NUM_SPLIT = 3
MAX_RES = 512
NUM = 3


def load_target_model(
        fp8_base: bool,
        pretrained_model_name_or_path: str,
        disable_mmap_load_safetensors: bool,
        clip_l_path: str,
        fp8_base_unet: bool,
        t5xxl_path: str,
        ae_path: str,
        weight_dtype: torch.dtype,
        accelerator: Accelerator
):
    loading_dtype = None if fp8_base else weight_dtype

    _, model = flux_utils.load_flow_model(
        pretrained_model_name_or_path,
        torch.float8_e4m3fn,
        accelerator.device,
        disable_mmap=disable_mmap_load_safetensors
    )

    if fp8_base:
        if model.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
        elif model.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 FLUX model")

    clip_l = flux_utils.load_clip_l(
        clip_l_path,
        weight_dtype,
        accelerator.device,
        disable_mmap=disable_mmap_load_safetensors
    )
    clip_l.eval()

    if fp8_base and not fp8_base_unet:
        loading_dtype_t5xxl = None
    else:
        loading_dtype_t5xxl = weight_dtype

    t5xxl = flux_utils.load_t5xxl(
        t5xxl_path,
        loading_dtype_t5xxl,
        accelerator.device,
        disable_mmap=disable_mmap_load_safetensors
    )
    t5xxl.eval()

    if fp8_base and not fp8_base_unet:
        if t5xxl.dtype in {torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz}:
            raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
        elif t5xxl.dtype == torch.float8_e4m3fn:
            logger.info("Loaded fp8 T5XXL model")

    ae = flux_utils.load_ae(
        ae_path,
        weight_dtype,
        accelerator.device,
        disable_mmap=disable_mmap_load_safetensors
    )

    return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model


def sample(accelerator, vae, text_encoder, flux):
    def encode_images_to_latents(vae, images):
        latents = vae.encode(images)
        return latents

    def concatenate_images(image1, image2):
        new_image = Image.new('RGB', (image1.width + image2.width, image1.height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))
        return new_image

    def resize_and_center_crop(image, target_width, target_height):
        original_width, original_height = image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return cropped_image

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    test_names = os.listdir(PROMPT_DIR)
    with torch.no_grad(), accelerator.autocast():
        for idx, name in enumerate(test_names):
            print(f"current idx: {idx}, name: {name}")
            portrait_name = name.split('_')[0] + '.png'
            background_name = name.split('_')[1].split('.')[0] + '.png'
            foreground = Image.open(FOREGROUND_DIR + portrait_name)
            w, h = foreground.size
            scale = max(w, h) / MAX_RES
            resized_w, resized_h = math.floor(w / scale / 64) * 64, math.floor(h / scale / 64) * 64
            foreground = foreground.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            background = Image.open(BACKGROUND_DIR + background_name)
            background = resize_and_center_crop(background, w, h)
            background = background.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            two_panel = concatenate_images(foreground, background)
            width, height = two_panel.size
            with open(PROMPT_DIR + name, 'r', encoding='utf-8') as f:
                prompt = f.read()
            for i in range(NUM):
                prompt_dict = {}
                prompt_dict["prompt"] = prompt
                prompt_dict['name'] = name.split('.')[0]+'+'+str(i) + '.png'
                print(f"generate: {prompt_dict['name']}")
                prompt_dict['width'] = width // 2 * 3
                prompt_dict['height'] = height

                image = img_transforms(np.array(two_panel, dtype=np.uint8)).unsqueeze(0).to(
                    vae.device,
                    dtype=vae.dtype
                )
                latents = encode_images_to_latents(vae, image)
                logger.info(f"Encoded latents shape: {latents.shape}")
                logger.info(f"Text Encoder outputs for prompt: {prompt}")

                sample_prompts_te_outputs = {}
                text_encoder[0].to(accelerator.device)
                text_encoder[1].to(accelerator.device)
                tokenize_strategy = strategy_flux.FluxTokenizeStrategy(512)
                text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(True)
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                    if p not in sample_prompts_te_outputs:
                        tokens_and_masks = tokenize_strategy.tokenize(p)
                        sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, text_encoder, tokens_and_masks, True
                        )
                sample_image_inference(
                    accelerator,
                    flux,
                    text_encoder,
                    vae,
                    prompt_dict,
                    sample_prompts_te_outputs,
                    None,
                    latents
                )
        clean_memory_on_device(accelerator.device)


def sample_image_inference(
        accelerator: Accelerator,
        flux: flux_models.Flux,
        text_encoder,
        ae: flux_models.AutoEncoder,
        prompt_dict,
        sample_prompts_te_outputs,
        prompt_replacement,
        lantents
):

    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict['width']
    height = prompt_dict['height']
    scale = prompt_dict.get("scale", 1.0)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        torch.cuda.seed()

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)
    logger.info(f"generate:{prompt_dict['name']}, width: {width}, height: {height}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    text_encoder_conds = []
    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        text_encoder_conds = sample_prompts_te_outputs[prompt]
    ae_outputs = lantents

    l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds

    logger.debug(
        f"l_pooled shape: {l_pooled.shape}, t5_out shape: {t5_out.shape}, txt_ids shape: {txt_ids.shape}, t5_attn_mask shape: {t5_attn_mask.shape}")

    weight_dtype = ae.dtype  # TODO: give dtype as argument
    packed_latent_height = height // 16
    packed_latent_width = width // 16

    logger.debug(f"packed_latent_height: {packed_latent_height}, packed_latent_width: {packed_latent_width}")

    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )

    timesteps = flux_train_utils.get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(
        accelerator.device, dtype=weight_dtype
    )
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    for param in flux.parameters():
        param.requires_grad = False

    with accelerator.autocast(), torch.no_grad():
        x = flux_train_utils.denoise(flux, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps,
                                     guidance=scale, t5_attn_mask=t5_attn_mask, ae_outputs=ae_outputs)

    logger.debug(f"x shape after denoise: {x.shape}")

    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])
    img_filename = prompt_dict.get("name")
    w, h = image.size
    cropped_image = image.crop((w // 3 * 2, 0, w, h))
    cropped_image.save(os.path.join(STORE_DIR, img_filename))


def main():
    accelerator = Accelerator(mixed_precision='bf16', device_placement=True)

    _, [clip_l, t5xxl], ae, model = load_target_model(
        fp8_base=True,
        pretrained_model_name_or_path=BASE_FLUX_CHECKPOINT,
        disable_mmap_load_safetensors=False,
        clip_l_path=CLIP_L_PATH,
        fp8_base_unet=False,
        t5xxl_path=T5XXL_PATH,
        ae_path=AE_PATH,
        weight_dtype=torch.bfloat16,
        accelerator=accelerator
    )

    model.eval()
    clip_l.eval()
    t5xxl.eval()
    ae.eval()

    multiplier = 1.0
    weights_sd = load_file(LORA_WEIGHTS_PATH)
    lora_model, _ = lora_flux.create_network_from_weights(multiplier, None, ae, [clip_l, t5xxl], model, weights_sd,
                                                          True)

    lora_model.apply_to([clip_l, t5xxl], model)
    info = lora_model.load_state_dict(weights_sd, strict=True)
    logger.info(f"Loaded LoRA weights from {LORA_WEIGHTS_PATH}: {info}")
    lora_model.eval()
    lora_model.to("cuda")

    text_encoder = [clip_l, t5xxl]

    sample(accelerator, vae=ae, text_encoder=text_encoder, flux=model)


if __name__ == "__main__":
    BASE_FLUX_CHECKPOINT = ""
    LORA_WEIGHTS_PATH = ""
    CLIP_L_PATH = ""
    T5XXL_PATH = ""
    AE_PATH = ""

    FOREGROUND_DIR = ""
    BACKGROUND_DIR = ""
    PROMPT_DIR = ""
    STORE_DIR = ""
    os.makedirs(STORE_DIR, exist_ok=True)
    main()
