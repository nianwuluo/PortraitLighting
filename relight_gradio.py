import matplotlib

matplotlib.use('Agg')
from PIL import Image
import torch
import numpy as np
import gradio as gr
from library.device_utils import clean_memory_on_device
from library import flux_models, flux_train_utils_relight as flux_train_utils, flux_utils, sd3_train_utils, \
    strategy_base, strategy_flux, train_util
from library.utils import setup_logging
import torchvision.transforms as transforms
import random
from accelerate import Accelerator

setup_logging()
import logging

logger = logging.getLogger(__name__)
accelerator = Accelerator(mixed_precision='bf16', device_placement=True)

RELIGHT_MODEL_PATH = {'Portrait Relight': 'models/relight-flux-14800/relight-flux-14800-merge.safetensors',
                      'Portrait Personalize': 'models/customize-flux-13380/customize-flux-13380-merge.safetensors'}
CLIP_L_PATH = "models/text_encoders/clip_l.safetensors"
T5XXL_PATH = "models/text_encoders/t5xxl_fp16.safetensors"
AE_PATH = "models/ae.safetensors"
device = torch.device('cuda')


def load_target_model(model_type: str):
    clean_memory_on_device(device)
    global model, clip_l, t5xxl, ae
    logger.info("Loading models...")
    try:
        _, model = flux_utils.load_flow_model(
            RELIGHT_MODEL_PATH[model_type], torch.float8_e4m3fn, device, disable_mmap=False
        )
        model.eval()
        clip_l = flux_utils.load_clip_l(CLIP_L_PATH, torch.bfloat16, device, disable_mmap=False)
        clip_l.eval()
        t5xxl = flux_utils.load_t5xxl(T5XXL_PATH, torch.bfloat16, device, disable_mmap=False)
        t5xxl.eval()
        ae = flux_utils.load_ae(AE_PATH, torch.bfloat16, device, disable_mmap=False)
        ae.eval()
        LOAD_MODEL = True
        return "Models loaded successfully."
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return f"Error loading models: {e}"


def generate(portrait_image, background_image, prompt, image_width, image_height, seed, steps, guidance):
    global model, clip_l, t5xxl, ae
    # if not LOAD_MODEL:
    #     gr.Warning("Load model first!")
    #     return
    logger.info(f"Using seed: {seed}")

    def concatenate_images(image1, image2):
        # 水平拼接
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
    foreground = portrait_image.resize((image_width, image_height), Image.Resampling.LANCZOS)
    background = resize_and_center_crop(background_image, image_width, image_height)
    background = background.resize((image_width, image_height), Image.Resampling.LANCZOS)
    two_panel = concatenate_images(foreground, background)
    width, height = two_panel.size
    image = img_transforms(np.array(two_panel, dtype=np.uint8)).unsqueeze(0).to(
        device,
        dtype=torch.bfloat16
    )
    ae.to(device)
    latents = ae.encode(image)
    logger.info("Image encoded to latents.")

    conditions = {}
    conditions[prompt] = latents
    clip_l.to(device)
    t5xxl.to(device)
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(512)
    text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(True)
    tokens_and_masks = tokenize_strategy.tokenize(prompt)
    l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, t5xxl],
                                                                                   tokens_and_masks, True)
    logger.info("Prompt encoded.")
    width = width // 2 * 3
    height = height

    if seed == -1:
        seed = random.randint(0, 100000000)

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    packed_latent_height = height // 16
    packed_latent_width = width // 16

    noise = torch.randn(1, packed_latent_height * packed_latent_width, 16 * 2 * 2, device=device, dtype=torch.float16,
                        generator=torch.Generator(device=device).manual_seed(seed))
    logger.info("Noise prepared.")
    timesteps = flux_train_utils.get_schedule(steps, noise.shape[1], shift=True)  # Sample steps = 20
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(device)

    t5_attn_mask = t5_attn_mask.to(device)
    ae_outputs = conditions[prompt]
    # Run the denoising process

    model.to(device)
    logger.info(f"Model device: {model.device}")
    logger.info(f"Noise device: {noise.device}")
    logger.info(f"Image IDs device: {img_ids.device}")
    logger.info(f"T5 output device: {t5_out.device}")
    logger.info(f"Text IDs device: {txt_ids.device}")
    logger.info(f"L pooled device: {l_pooled.device}")
    with accelerator.autocast(), torch.no_grad():
        x = flux_train_utils.denoise(
            model, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance,
            t5_attn_mask=t5_attn_mask, ae_outputs=ae_outputs
        )
    logger.info("Denoising process completed.")
    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    logger.info("Latents decoded into image.")
    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])
    logger.info("Image generation completed.")
    w, h = image.size
    cropped_image = image.crop((w // 3 * 2, 0, w, h))
    return cropped_image


def relight_portrait_based_background():
    LOAD_MODEL = False
    gr.Markdown("## 基于背景图像的人像光效生成 Background-based Portrait Relight")
    gr.HTML(relight_portrait_based_background_tips)
    with gr.Row():
        with gr.Column(scale=1):
            # Dropdown for selecting the recraft model
            relight_model = gr.Textbox(label="Model", placeholder="Portrait Relight", interactive=False,
                                       value="Portrait Relight", lines=1)
            # Load Model Button
            load_button = gr.Button("Load Model")

        with gr.Column(scale=1):
            # Status message box
            status_box = gr.Textbox(label="Status", placeholder="Model loading status", interactive=False,
                                    value="Model not loaded", lines=1)
    with gr.Row():
        with gr.Column(scale=2, min_width=100):
            with gr.Row():
                with gr.Column(scale=1):
                    portrait_image = gr.Image(type="pil", label="Portrait Image", height=360)

                with gr.Column(scale=1):
                    background_image = gr.Image(type="pil", label="Background Image", height=360)
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(value="A woman", label='Prompt', lines=2)
                with gr.Column(scale=1):
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=512, value=320, step=64)
                    with gr.Row():
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=512, value=512, step=64)
            with gr.Row():
                generate_button = gr.Button(value="Generate")
            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=20, maximum=50, value=30, step=1)
                guidance = gr.Slider(label="Guidance", minimum=0.5, maximum=6, value=1, step=0.5)
                seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=100000000, step=1,
                                 value=42)
        with gr.Column(scale=1, min_width=100):
            relight_image = gr.Image(label="Output", interactive=False, height=480)
    ips = [portrait_image, background_image, prompt, image_width, image_height, seed, steps, guidance]

    load_button.click(fn=load_target_model, inputs=[relight_model], outputs=[status_box])
    generate_button.click(fn=generate, inputs=ips, outputs=[relight_image])


def text_to_personalize_portrait():
    LOAD_MODEL = False
    gr.Markdown("## ID保持的个性化光照人像生成 Text to Portrait Personalize")
    gr.HTML(text_to_personalize_portrait_tips)
    with gr.Row():
        with gr.Column(scale=1):
            # Dropdown for selecting the recraft model
            customize_model = gr.Textbox(label="Model", placeholder="Portrait Personalize", interactive=False,
                                         value="Portrait Personalize", lines=1)
            # Load Model Button
            load_button = gr.Button("Load Model")

        with gr.Column(scale=1):
            # Status message box
            status_box = gr.Textbox(label="Status", placeholder="Model loading status", interactive=False,
                                    value="Model not loaded", lines=1)
    with gr.Row():
        with gr.Column(scale=2, min_width=100):
            with gr.Row():
                with gr.Column(scale=1):
                    face_image = gr.Image(type="pil", label="Face Image", height=360)

                with gr.Column(scale=1):
                    background_image = gr.Image(type="pil", label="Background Image", height=360)
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(value="A woman", label='Prompt', lines=2)
                with gr.Column(scale=1):
                    with gr.Row():
                        image_width = gr.Slider(label="Image Width", minimum=256, maximum=512, value=512, step=64)
                    with gr.Row():
                        image_height = gr.Slider(label="Image Height", minimum=256, maximum=512, value=512, step=64)
            with gr.Row():
                generate_button = gr.Button(value="Generate")
            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=20, maximum=50, value=30, step=1)
                guidance = gr.Slider(label="Guidance", minimum=0.5, maximum=6, value=3.5, step=0.5)
                seed = gr.Slider(label="Seed (-1 indicates random)", minimum=-1, maximum=100000000, step=1,
                                 value=26)
        with gr.Column(scale=1, min_width=100):
            personalize_image = gr.Image(label="Output", interactive=False, height=480)
    ips = [face_image, background_image, prompt, image_width, image_height, seed, steps, guidance]

    load_button.click(fn=load_target_model, inputs=[customize_model], outputs=[status_box])
    generate_button.click(fn=generate, inputs=ips, outputs=[personalize_image])


if __name__ == "__main__":
    LOAD_MODEL = False
    relight_portrait_based_background_tips = r"""🚀🚀🚀Quick start:
    <p> 1. Click the <b>Load Model</b> button to load model weights. </p> 
    <p> 2. Upload a portrait image and a background image. </p>
    <p> 3. Enter a brief prompt (English). Guidance set to 1. </p>
    <p> 4. Click the <b>Generate</b> button to generate re-lit portrait. 🤗 </p> """
    text_to_personalize_portrait_tips = r"""🚀🚀🚀Quick start:
    <p> 1. Click the <b>Load Model</b> button to load model weights. </p> 
    <p> 2. Upload a face image and a background image. </p>
    <p> 3. Enter a prompt (English) to describe the portrait (clothing, hair styles, posture). Guidance set to 3.5 . </p>
    <p> 4. Click the <b>Generate</b> button to generate personalization lighting portrait. 🤗 </p> """
    block = gr.Blocks(title="PortraitLighting").queue()
    with block:
        with gr.TabItem("Relight Portrait"):
            relight_portrait_based_background()

        with gr.TabItem("Personalize Portrait"):
            text_to_personalize_portrait()

    block.launch(server_name='0.0.0.0', server_port=6006)
