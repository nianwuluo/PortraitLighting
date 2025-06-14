import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: 128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import math
import random
from typing import Any, Optional, Union


from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils,
    strategy_base,
    strategy_flux,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_schnell: Optional[bool] = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
                          val_dataset_group: Optional[train_util.DatasetGroup]):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.fp8_base_unet:
            args.fp8_base = True

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        self.train_clip_l = not args.network_train_unet_only
        self.train_t5xxl = False

        if args.max_token_length is not None:
            logger.warning("max_token_length is not used in Flux training / max_token_lengthはFluxのトレーニングでは使用されません")

        assert (
                       args.blocks_to_swap is None or args.blocks_to_swap == 0
               ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"

        if args.split_mode:
            if args.blocks_to_swap is not None:
                logger.warning(
                    "split_mode is deprecated. Because `--blocks_to_swap` is set, `--split_mode` is ignored."
                    " / split_modeは非推奨です。`--blocks_to_swap`が設定されているため、`--split_mode`は無視されます。"
                )
            else:
                logger.warning(
                    "split_mode is deprecated. Please use `--blocks_to_swap` instead. `--blocks_to_swap 18` is automatically set."
                    " / split_modeは非推奨です。代わりに`--blocks_to_swap`を使用してください。`--blocks_to_swap 18`が自動的に設定されました。"
                )
                args.blocks_to_swap = 18

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def load_target_model(self, args, weight_dtype, accelerator):

        loading_dtype = None if args.fp8_base else weight_dtype

        self.is_schnell, model = flux_utils.load_flow_model(
            args.pretrained_model_name_or_path, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        if args.fp8_base:
            # check dtype of model
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 FLUX model")
            else:
                logger.info(
                    "Cast FLUX model to fp8. This may take a while. You can reduce the time by using fp8 checkpoint."
                    " / FLUXモデルをfp8に変換しています。これには時間がかかる場合があります。fp8チェックポイントを使用することで時間を短縮できます。"
                )
                model.to(torch.float8_e4m3fn)


        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu",
                                        disable_mmap=args.disable_mmap_load_safetensors)
        clip_l.eval()

        if args.fp8_base and not args.fp8_base_unet:
            loading_dtype = None  # as is
        else:
            loading_dtype = weight_dtype

        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            # check dtype of model
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")

        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)

        return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model

    def get_tokenize_strategy(self, args):
        _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

        if args.t5xxl_max_token_length is None:
            if is_schnell:
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        return strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk,
                                                                            args.vae_batch_size, False)
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        self.train_t5xxl = network.train_t5xxl

        if self.train_t5xxl and args.cache_text_encoder_outputs:
            raise ValueError(
                "T5XXL is trained, so cache_text_encoder_outputs cannot be used / T5XXL学習時はcache_text_encoder_outputsは使用できません"
            )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            if self.train_clip_l and not self.train_t5xxl:
                return text_encoders[0:1]
            else:
                return None
        else:
            return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_clip_l, self.train_t5xxl]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_clip_l or self.train_t5xxl,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
            self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup,
            weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device)

            if text_encoders[1].dtype == torch.float8_e4m3fn:
                self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
            else:
                text_encoders[1].to(weight_dtype)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            if not self.is_train_text_encoder(args):
                logger.info("move CLIP-L back to cpu")
                text_encoders[0].to("cpu")
            logger.info("move t5XXL back to cpu")
            text_encoders[1].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device)


    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, flux):
        text_encoders = text_encoder
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoders)

        flux_train_utils.sample_images(
            accelerator, args, epoch, global_step, flux, ae, text_encoders, self.sample_prompts_te_outputs
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
                                                                          shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return vae.encode(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
            self,
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet: flux_models.Flux,
            network,
            weight_dtype,
            train_unet,
            is_train=True
    ):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )

        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(
            device=accelerator.device)

        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
        if not args.apply_t5_attn_mask:
            t5_attn_mask = None

        def call_dit(img, img_ids, t5_out, txt_ids, l_pooled, timesteps, guidance_vec, t5_attn_mask):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                model_pred = unet(
                    img=img,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                    txt_attention_mask=t5_attn_mask,
                )
            return model_pred

        model_pred = call_dit(
            img=packed_noisy_model_input,
            img_ids=img_ids,
            t5_out=t5_out,
            txt_ids=txt_ids,
            l_pooled=l_pooled,
            timesteps=timesteps,
            guidance_vec=guidance_vec,
            t5_attn_mask=t5_attn_mask,
        )

        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input,
                                                                             sigmas)

        target = noise - latents

        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                unet.prepare_block_swap_before_forward()
                with torch.no_grad():
                    model_pred_prior = call_dit(
                        img=packed_noisy_model_input[diff_output_pr_indices],
                        img_ids=img_ids[diff_output_pr_indices],
                        t5_out=t5_out[diff_output_pr_indices],
                        txt_ids=txt_ids[diff_output_pr_indices],
                        l_pooled=l_pooled[diff_output_pr_indices],
                        timesteps=timesteps[diff_output_pr_indices],
                        guidance_vec=guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None,
                        t5_attn_mask=t5_attn_mask[diff_output_pr_indices] if t5_attn_mask is not None else None,
                    )
                network.set_multiplier(1.0)

                model_pred_prior = flux_utils.unpack_latents(model_pred_prior, packed_latent_height,
                                                             packed_latent_width)
                model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")

    def update_metadata(self, metadata, args):
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        if index == 0:
            return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)
        else:
            text_encoder.encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        if index == 0:
            logger.info(f"prepare CLIP-L for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}")
            text_encoder.to(te_weight_dtype)
            text_encoder.text_model.embeddings.to(dtype=weight_dtype)
        else:
            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)
                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        module.forward = forward_hook(module)

            if flux_utils.get_t5xxl_actual_dtype(
                    text_encoder) == torch.float8_e4m3fn and text_encoder.dtype == weight_dtype:
                logger.info(f"T5XXL already prepared for fp8")
            else:
                logger.info(
                    f"prepare T5XXL for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}, add hooks")
                text_encoder.to(te_weight_dtype)
                prepare_fp8(text_encoder, weight_dtype)

    def prepare_unet_with_accelerator(
            self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        flux: flux_models.Flux = unet
        flux = accelerator.prepare(flux, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(flux).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(flux).prepare_block_swap_before_forward()
        return flux


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--split_mode",
        action="store_true",
        help="[Deprecated] This option is deprecated. Please use `--blocks_to_swap` instead."
             " / このオプションは非推奨です。代わりに`--blocks_to_swap`を使用してください。",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)
