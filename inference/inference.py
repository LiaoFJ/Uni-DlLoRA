#!anaconda {YOUR_ENV} python
# -*- coding: UTF-8 -*-
'''
@Project ：Uni-DlLoRA 
@File    ：inference.py
@IDE     ：PyCharm 
@Author  ：Fangjian LIAO
@Date    ：2024/11/14 22:56 
@Description: simple python script for inference
'''
import argparse
import os
import torch

import numpy as np
from PIL import Image
from diffusers import T2IAdapter, AutoencoderKL, DDPMScheduler, MultiAdapter
from controlnet_aux.pidi import PidiNetDetector

from custom_module.custom_unet import CustomUNet2DConditionModel
from diffusion_pipeline.pretrained.custom_adapter import StableDiffusionAdapterImg2ImgPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument("--sd_path", type=str, default='KiwiXR/stable-diffusion-v1-5',
                        required=False, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--checkpoint_save_dir", type=str, default="E:\\workspace\\checkpoint\\uniadapter_duallora",
                        required=False, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--input_image", type=str, default="../test_image/input_image.jpg",
                        required=False, help="Path of input image")
    parser.add_argument("--image_save_dir", type=str, default="../generated",
                        required=False, help="Path to save the generated image")
    args = parser.parse_args()
    return args


def generate(args):
    # load adapter
    adapter = MultiAdapter.from_pretrained(os.path.join(args.checkpoint_save_dir, "adapter"), torch_dtype=torch.float32).to("cuda")

    # load pidnet
    pidnet = PidiNetDetector.from_pretrained(os.path.join(args.checkpoint_save_dir, "pidnet")).to("cuda")

    # load euler_a scheduler
    ddpm = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    unet = CustomUNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet", torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder='vae', torch_dtype=torch.float32)

    # load lora
    unet.load_duallora_weight(args.checkpoint_save_dir, subfolder="custom_duallora")
    pipe = StableDiffusionAdapterImg2ImgPipeline.from_pretrained(
        args.sd_path, vae=vae, unet=unet, adapter=adapter, scheduler=ddpm, torch_dtype=torch.float32).to("cuda")

    # load lora into unet: Actually can use pop()
    idx = 0
    for attn_processor_name, attn_processor in pipe.unet.attn_processors.items():
        # Parse the attention module.
        attn_module = pipe.unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)

        # Set the `lora_layer` attribute of the attention-related matrices.
        attn_module.to_q.set_lora_layer(pipe.unet.extra_lora_parameters['illu_lora_parameter'][idx])
        attn_module.to_k.set_lora_layer(pipe.unet.extra_lora_parameters['illu_lora_parameter'][idx + 1])
        attn_module.to_v.set_lora_layer(pipe.unet.extra_lora_parameters['illu_lora_parameter'][idx + 2])
        attn_module.to_out[0].set_lora_layer(pipe.unet.extra_lora_parameters['illu_lora_parameter'][idx + 3])
        idx += 4

    if not os.path.exists(args.image_save_dir):
        os.makedirs(args.image_save_dir)
    # generate image
    generator = torch.Generator(device="cuda").manual_seed(28)

    image = Image.open(args.input_image)
    sketch = pidnet(input_image=image, detect_resolution=512, image_resolution=512,
                    apply_filter=False).convert('L')

    gen_images = pipe(
        prompt='',
        image=image,
        adapter_image=[image, sketch],
        num_inference_steps=50,
        adapter_conditioning_scale=1,
        guidance_scale=5,
        strength=0.7,
        generator=generator,
    ).images[0]
    gen_images.save(os.path.join(args.image_save_dir, "generated.png"))
    return


if __name__ == '__main__':
    args = parse_args()
    generate(args)
