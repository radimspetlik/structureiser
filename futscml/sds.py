from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import retrieve_timesteps

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm


class SDSControlNet(nn.Module):
    def __init__(
            self,
            device,
            fp16=True,
            vram_O=False,
            sd_version="1.5",
            hf_key=None,
            t_range=[0.02, 0.98],
            checkpoint="ControlNet-1-1-preview/control_v11p_sd15_lineart"
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.bfloat16 if fp16 else torch.float32

        # Create model
        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=self.dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_key, controlnet=controlnet, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler", torch_dtype=self.dtype
        # )
        self.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
            self,
            pred_rgb,
            control_image,
            step_ratio=None,
            guidance_scale=100,
            as_latent=False,
            vers=None, hors=None,
            epoch=None,
            inference_step=27,  # out of 30, the step 29 is the t=0
            skip_interpolation=False,
            return_image=False,
            use_adaptive_mask=False,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        control_image = control_image.to(self.dtype)

        if not skip_interpolation:
            control_image = F.interpolate(control_image, (512, 512), mode="bilinear", align_corners=False)

        control_image_0_1 = control_image.clone()
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = pred_rgb
            if not skip_interpolation:
                # interp to 512x512 to be fed into vae.
                pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
            latents_input = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            num_inference_steps = 30
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, None)
            # print(timesteps)

            t_idx = inference_step
            t = timesteps[t_idx:t_idx+1]
            t = repeat(t, '1 -> b', b=batch_size)
            self.scheduler._step_index = t_idx
            timesteps = timesteps[t_idx:t_idx+1]

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1),
                                    self.embeddings['neg'].expand(batch_size, -1, -1)])

            enumerator = enumerate(timesteps)
            if len(timesteps) > 1:
                enumerator = enumerate(tqdm(timesteps))

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            for i, t in enumerator:
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                control_image = torch.cat([control_image] * 2)
                # tt = torch.cat([t] * 2)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embeddings,
                    controlnet_cond=control_image,
                    conditioning_scale=1.0,
                    guess_mode=False,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform classifier_free_guidance (high scale from paper!)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, return_dict=False)[0]

                grad = w * (noise_pred - noise)
                grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

            if return_image:
            # if epoch is not None or True:
            #     epoch = 0
            #     imgs_in = self.decode_latents(latents_input)
                latents = self.scheduler.convert_model_output(noise_pred, sample=latents_noisy)
                imgs = self.decode_latents(latents)
            #     # # Img to Numpy
            #     imgs_in = imgs_in.detach().cpu().permute(0, 2, 3, 1).numpy()
            #     imgs_in = (imgs_in * 255).round().astype("uint8")[..., ::-1].copy()
            #     imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            #     imgs = (imgs * 255).round().astype("uint8")[..., ::-1].copy()
            #     import cv2, numpy as np
            #     cv2.imwrite(f'image_out_{epoch:05d}.png', np.concatenate((imgs_in[0], imgs[0]), axis=1))
            #     exit()
                return imgs

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]
        if use_adaptive_mask:
            control_image_0_1 = control_image_0_1.mean(dim=1, keepdim=True)
            control_image_0_1 = F.interpolate(control_image_0_1, (target.shape[-2], target.shape[-1]), mode="bilinear", align_corners=False)
            loss = 0.5 * F.mse_loss(latents.float() * control_image_0_1, target * control_image_0_1, reduction='sum') / latents.shape[0]

        return loss

    def train_step_lmc(
            self,
            pred_rgb,
            control_image,
            step_ratio=None,
            guidance_scale=100,
            as_latent=False,
            vers=None, hors=None,
            epoch=None,
            inference_step=27,  # out of 30, the step 29 is the t=0
            skip_interpolation=False,
            use_adaptive_mask=False,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        control_image = control_image.to(self.dtype)

        if not skip_interpolation:
            control_image = F.interpolate(control_image, (512, 512), mode="bilinear", align_corners=False)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            pred_rgb_512 = pred_rgb
            if not skip_interpolation:
                # interp to 512x512 to be fed into vae.
                pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)
            latents_input = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            num_inference_steps = 30
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, None)
            # print(timesteps)

            t_idx = inference_step
            t = timesteps[t_idx:t_idx+1]
            t = repeat(t, '1 -> b', b=batch_size)
            self.scheduler._step_index = t_idx
            timesteps = timesteps[t_idx:t_idx+1]

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1),
                                    self.embeddings['neg'].expand(batch_size, -1, -1)])

            enumerator = enumerate(timesteps)
            if len(timesteps) > 1:
                enumerator = enumerate(tqdm(timesteps))

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            for i, t in enumerator:
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                control_image = torch.cat([control_image] * 2)
                # tt = torch.cat([t] * 2)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embeddings,
                    controlnet_cond=control_image,
                    conditioning_scale=1.0,
                    guess_mode=False,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform classifier_free_guidance (high scale from paper!)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy, return_dict=False)[0]

            # seems important to avoid NaN...

            latents_hat = self.scheduler.convert_model_output(noise_pred, sample=latents_noisy)

            # if epoch is not None or True:
            #     epoch = 0
            #     imgs_in = self.decode_latents(latents_input)
            #     latents = self.scheduler.convert_model_output(noise_pred, sample=latents_noisy)
            #     imgs = self.decode_latents(latents)
            #     # # Img to Numpy
            #     imgs_in = imgs_in.detach().cpu().permute(0, 2, 3, 1).numpy()
            #     imgs_in = (imgs_in * 255).round().astype("uint8")[..., ::-1].copy()
            #     imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
            #     imgs = (imgs * 255).round().astype("uint8")[..., ::-1].copy()
            #     import cv2, numpy as np
            #     cv2.imwrite(f'image_out_{epoch:05d}.png', np.concatenate((imgs_in[0], imgs[0]), axis=1))
            #     exit()

        # target = (latents - grad).detach()
        loss = F.mse_loss(latents_hat.float(), latents, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(
            self,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat(
            [self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        imgs = self.decode_latents_m1p1(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def decode_latents_m1p1(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample

        return imgs

    def encode_imgs_m1p1(self, imgs_m1p1):
        posterior = self.vae.encode(imgs_m1p1).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        return self.encode_imgs_m1p1(imgs)

    def prompt_to_img(
            self,
            prompts,
            negative_prompts="",
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)

        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs