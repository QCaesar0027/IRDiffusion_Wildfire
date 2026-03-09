#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 ControlNet wildfire generation baseline vs +UNet LoRA
"""
import os
import json
import argparse
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time

import torch
from torchvision import transforms


from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)

# ===============================
# Generator
# ===============================

class FlameGenerator:
    def __init__(
        self,
        base_model_path = "runwayml/stable-diffusion-v1-5",
        device = "cuda",
        pretrained_controlnet_id = "lllyasviel/control_v11p_sd15_inpaint",
       
        lora_path = None,
        lora_scale = 1.0,
        fuse_lora = False,
    ):
        self.device = device
        self.base_model_path = base_model_path
        self.pretrained_controlnet_id = pretrained_controlnet_id

        self.lora_path = lora_path
        self.lora_scale = float(lora_scale)
        self.fuse_lora = bool(fuse_lora)

        print(f"Using pretrained ControlNet {pretrained_controlnet_id}")
        print(f"Loading base model {base_model_path}")

       
        try:
            self.controlnet = ControlNetModel.from_pretrained(
                pretrained_controlnet_id, torch_dtype=torch.float16
            )
            print("Pretrained ControlNet model loaded successfully")
            total_params = sum(p.numel() for p in self.controlnet.parameters())
            trainable_params = sum(
                p.numel() for p in self.controlnet.parameters() if p.requires_grad
            )
            print("Pretrained ControlNet parameter statistics")
            print(f"   Total parameters {total_params:,}")
            print(f"   Trainable parameters {trainable_params:,}")
        except Exception as e:
            print(f"Failed to load pretrained ControlNet {e}")
            raise

       
        try:
            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                base_model_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception as e:
            print(f"Failed to load using model ID {e}")
            if os.path.exists(base_model_path):
                print(f"Attempting to load from local path {base_model_path}")
                self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    base_model_path,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                raise Exception(
                    f"Failed to load base model please check the path or ensure the model is cached {base_model_path}"
                )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        self.pipe.enable_attention_slicing()  # Save VRAM
        print("Model loaded successfully")

        # Add LoRA to UNet optional
        if self.lora_path is not None:
            self._attach_unet_lora(self.lora_path, self.lora_scale, self.fuse_lora)

    # -------------------------------
    # Control map construction and detail functions
    # -------------------------------
    def compute_gradient_info(self, ir_tensor):
        import torch.nn.functional as F

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=ir_tensor.dtype,
            device=ir_tensor.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=ir_tensor.dtype,
            device=ir_tensor.device,
        ).view(1, 1, 3, 3)

        if len(ir_tensor.shape) == 2:
            ir_input = ir_tensor.unsqueeze(0).unsqueeze(0)
            need_squeeze = True
        elif len(ir_tensor.shape) == 3:
            ir_input = ir_tensor.unsqueeze(0)
            need_squeeze = True
        else:
            ir_input = ir_tensor
            need_squeeze = False

        if ir_input.shape[1] > 1:
            grad_x_list, grad_y_list = [], []
            for c in range(ir_input.shape[1]):
                single = ir_input[:, c : c + 1, :, :]
                grad_x_c = F.conv2d(single, sobel_x, padding=1)
                grad_y_c = F.conv2d(single, sobel_y, padding=1)
                grad_x_list.append(grad_x_c)
                grad_y_list.append(grad_y_c)
            grad_x = torch.cat(grad_x_list, dim=1)
            grad_y = torch.cat(grad_y_list, dim=1)
        else:
            grad_x = F.conv2d(ir_input, sobel_x, padding=1)
            grad_y = F.conv2d(ir_input, sobel_y, padding=1)

        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        gradient_magnitude = torch.pow(gradient_magnitude, 0.7)
        gradient_magnitude = gradient_magnitude.clamp(0.0, 1.0)
    
        if need_squeeze:
            gradient_magnitude = gradient_magnitude.squeeze(0)
        return gradient_magnitude

    def _attach_unet_lora(self, lora_path, lora_scale = 1.0, fuse = False):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path does not exist {lora_path}")

        print(f"LoRA mounted only to UNet {lora_path}")

        # Compatible with passing file or directory
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path) or "."
            weight_name = os.path.basename(lora_path)
        else:
            lora_dir = lora_path
            weight_name = None

        # Only inject LoRA into UNet will throw an error if weights are not for UNet
        try:
            self.pipe.unet.load_attn_procs(lora_dir, weight_name=weight_name)
            print("LoRA loaded into UNet unet.load_attn_procs")
        except Exception as e:
            raise RuntimeError(
                "Failed to load into UNet looks like this LoRA is not a UNet LoRA or weight naming is incompatible\n"
                "   Solution please use another UNet LoRA weight or specify as UNet when exporting training\n"
                f"   Original exception {e}"
            )

        # Set LoRA scale some versions do not support dynamic adjustment failure can be ignored
        try:
            self.pipe.set_adapters(["default"], [float(lora_scale)])
            print(f"LoRA scale set to {lora_scale}")
        except Exception:
            if abs(lora_scale - 1.0) > 1e-6:
                print("Current diffusers does not support dynamic set_adapters scale remains 1.0")

        # If fusion is needed it only affects UNet
        if fuse:
            try:
                self.pipe.fuse_lora(lora_scale=float(lora_scale))
                print("LoRA fused into UNet scale cannot be adjusted or switched afterwards")
            except Exception as e:
                print(f"Fusion failed possibly unsupported version ignored {e}")
    
    def prepare_control_hint(
        self,
        ir_image_path,
        resolution=512,
        q_high=0.95,
        q_low=0.05,
        morph_kernel=3,
        local_kernel=5,
        local_thresh=0.2,
    ):
        ir_image = Image.open(ir_image_path).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.ToTensor(),
            ]
        )
        ir_tensor = transform(ir_image)  # [1, H, W] in [0,1]
        print(f"IR tensor shape {ir_tensor.shape} range {ir_tensor.min():.3f} {ir_tensor.max():.3f}")

        ir_small = torch.nn.functional.interpolate(ir_tensor.unsqueeze(0), scale_factor=0.25, mode="area").squeeze(0)
        flat = ir_small.flatten()

        if len(flat) > 0 and not torch.isnan(flat).all():
            valid_flat = flat[~torch.isnan(flat)]
            thr = torch.quantile(valid_flat, q_high) if len(valid_flat) > 0 else torch.tensor(0.5)
        else:
            valid_flat = flat
            thr = torch.tensor(0.5)
        # Ablation experiment 1
        thr = ir_tensor.mean()    

        print(f"Using threshold calculation from training q_high {q_high:.2f} threshold {thr:.3f}")

        soft_mask = torch.sigmoid((ir_tensor - thr) * 10)
        soft_mask = torch.nan_to_num(soft_mask, nan=0.0)
        print(f"Soft mask range {soft_mask.min():.3f} {soft_mask.max():.3f}")

        def _dilate(x, k=3):  return torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        def _erode(x, k=3):   return 1 - torch.nn.functional.max_pool2d(1 - x, kernel_size=k, stride=1, padding=k // 2)

        soft_mask = _dilate(_erode(soft_mask, morph_kernel), morph_kernel)

        pad = local_kernel // 2
        local_avg = torch.nn.functional.avg_pool2d(soft_mask, kernel_size=local_kernel, stride=1, padding=pad)
        soft_mask = (local_avg >= local_thresh).float() * soft_mask
        print(f"Soft mask range after morphological processing {soft_mask.min():.3f} {soft_mask.max():.3f}")

        if len(valid_flat) > 0:
            p_low = torch.quantile(valid_flat, q_low)
            p_high = torch.quantile(valid_flat, 0.99)
        else:
            p_low, p_high = torch.tensor(0.0), torch.tensor(1.0)

        print(f"Intensity range p_low {p_low:.3f} p_high {p_high:.3f}")


        # Ablation experiment 1
        p_low  = ir_tensor.min()
        p_high = ir_tensor.max()

        if p_high > p_low + 1e-6:
            intensity_map = (ir_tensor - p_low) / (p_high - p_low)
            intensity_map = intensity_map.clamp(0.0, 1.0)
            intensity_map = torch.pow(intensity_map, 0.7).clamp(0.0, 1.0)
        else:
            intensity_map = torch.zeros_like(ir_tensor)

        intensity_map = torch.nan_to_num(intensity_map, nan=0.0)
        # intensity_map = intensity_map * soft_mask
        print(f"Intensity map range {intensity_map.min():.3f} {intensity_map.max():.3f}")

        gradient_info = self.compute_gradient_info(ir_tensor)
        gradient_info = torch.nan_to_num(gradient_info, nan=0.0) * soft_mask
        if gradient_info.max() > gradient_info.min() + 1e-8:
            gradient_info = (gradient_info - gradient_info.min()) / (gradient_info.max() - gradient_info.min() + 1e-8)
            gradient_info = torch.pow(gradient_info, 0.8)
        gradient_info = torch.clamp(gradient_info, 0.0, 1.0)
        print(f"Gradient info range {gradient_info.min():.3f} {gradient_info.max():.3f}")

        controlnet_hint = torch.cat([soft_mask, intensity_map, gradient_info], dim=0)
        controlnet_hint = torch.nan_to_num(controlnet_hint, nan=0.0)
        print(f"Final control signal shape {controlnet_hint.shape} range {controlnet_hint.min():.3f} {controlnet_hint.max():.3f}")

        print("Diagnostic info")
        print(f"   IR original value range {ir_tensor.min():.3f} {ir_tensor.max():.3f}")
        print(f"   IR mean {ir_tensor.mean():.3f}")
        print(f"   Threshold {thr:.3f}")
        print(f"   Ratio of pixels above threshold {(ir_tensor > thr).float().mean():.3f}")
        sigmoid_input = (ir_tensor - thr) * 10
        print(f"   Sigmoid input range {sigmoid_input.min():.3f} {sigmoid_input.max():.3f}")
        print(f"   Ratio of Sigmoid input greater than 0 {(sigmoid_input > 0).float().mean():.3f}")

        return controlnet_hint

    # -------------------------------
    # Generation
    # -------------------------------
    def generate_flame(
        self,
        rgb_image_path,
        ir_image_path,
        prompt="A cinematic, high-resolution, ultra-detailed photograph of a realistic wildfire flame",
        negative_prompt="blue, cyan, purple, green, cold colors, sky, water, ice",
        num_inference_steps=20,
        guidance_scale=7.5,
        strength=0.8,
        seed=None,
        save_debug=False,
        mask_threshold=0.3,
        control_type="rgb",
    ):
        if seed is not None:
            torch.manual_seed(seed)

        print("Start processing image")
        print(f"   RGB image {rgb_image_path}")
        print(f"   IR image {ir_image_path}")

        rgb_image = Image.open(rgb_image_path).convert("RGB")
        control_hint = self.prepare_control_hint(
            ir_image_path,
            resolution=512,
            q_high=0.95,
            q_low=0.05,
            morph_kernel=3,
            local_kernel=5,
            local_thresh=0.2,
        )
        rgb_image = rgb_image.resize((512, 512))
        control_image_tensor = control_hint.permute(1, 2, 0)

        ir_raw = Image.open(ir_image_path).convert("L")
        ir_raw = ir_raw.resize((512, 512))


        gradient_channel_tensor = control_image_tensor[:, :, 2]

        control_image_np = control_hint.permute(1, 2, 0).numpy()
        
        # rgb_control_origin = np.stack(
        #     [
        #         control_image_np[:, :, 0] * 255,
        #         control_image_np[:, :, 1] * 255,
        #         control_image_np[:, :, 2] * 255,
        #     ],
        #     axis=2,
        # ).astype(np.uint8)
        soft_mask_channel = control_image_np[:, :, 0]
        intensity_map_channel = control_image_np[:, :, 1]
        # gradient_channel = control_image_np[:, :, 2]


       

        # Scheme 1 use intensity intensity squared zero as RGB control map
        soft_mask_channel_tensor = control_hint[0,:,:]
        intensity_map_tensor = control_hint[1, :, :]
      


        r_channel = intensity_map_tensor
        g_channel = torch.pow(intensity_map_tensor, 2)
        # r_channel = torch.pow(intensity_map_tensor, 2)

        # g_channel = intensity_map_tensor

        # g_channel = torch.zeros_like(intensity_map_tensor)
        # b_channel = torch.zeros_like(intensity_map_tensor)

        # g_channel = intensity_map_tensor
        b_channel = torch.zeros_like(intensity_map_tensor)


        # b_channel = gradient_channel_tensor

        new_controlnet_hint = torch.stack([r_channel, g_channel, b_channel], dim=0)

        control_image_np = new_controlnet_hint.permute(1, 2, 0).cpu().numpy()
        rgb_control = np.stack(
            [
                control_image_np[:, :, 0] * 255,
                control_image_np[:, :, 1] * 255,
                control_image_np[:, :, 2] * 255,
            ],
            axis=2,
        ).astype(np.uint8)

        # Scheme 2 weighted fusion optional
        # weights = [0.6, 0.3, 0.1]
        # fused_control = (
        #     weights[0] * soft_mask_channel
        #     + weights[1] * intensity_map_channel
        #     + weights[2] * gradient_channel
        # ) * 255

        if control_type == "rgb":
            control_image = Image.fromarray(rgb_control, mode="RGB")
            print("   Using RGB control image 3 channel info")
        elif control_type == "fused":
            control_image = Image.fromarray(fused_control.astype(np.uint8), mode="L")
            print("   Using weighted fusion control image")
        else:
            control_image = Image.fromarray(rgb_control_origin, mode="RGB")
            print("   Using single channel soft mask control image")

        # Mask
        soft_mask_for_mask = soft_mask_channel
        mask_binary = (soft_mask_for_mask > mask_threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_binary, mode="L")
        mask_coverage = np.sum(mask_binary > 0) / mask_binary.size * 100

        if mask_coverage > 0:
            from scipy import ndimage
            try:
                kernel = np.ones((3, 3), np.uint8)
                mask_array = np.array(mask_image)
                mask_array = ndimage.binary_dilation(mask_array > 0, structure=kernel).astype(np.uint8) * 255
                mask_image = Image.fromarray(mask_array, mode="L")
                print("Mask edge enhancement applied")
            except ImportError:
                print("Scipy not installed skipping mask edge enhancement")

        print(f"Mask coverage {mask_coverage:.1f} percent "
              f"{np.sum(mask_binary > 0)} {mask_binary.size} pixels")

        if mask_coverage > 30:
            print(f"Mask coverage is high {mask_coverage:.1f} percent attempting adaptive threshold")
            adaptive_threshold = np.percentile(soft_mask_for_mask, 95)
            mask_binary_adaptive = (soft_mask_for_mask > adaptive_threshold).astype(np.uint8) * 255
            adaptive_coverage = np.sum(mask_binary_adaptive > 0) / mask_binary_adaptive.size * 100
            print(f"   Adaptive threshold {adaptive_threshold:.3f} coverage {adaptive_coverage:.1f} percent")
            if adaptive_coverage < 30:
                mask_binary = mask_binary_adaptive
                mask_image = Image.fromarray(mask_binary, mode="L")
                mask_coverage = adaptive_coverage
                print("   Adaptive threshold applied")

        if mask_coverage < 1.0:
            print("Mask coverage is very low hot spots might not be detected consider lowering mask threshold")
        elif mask_coverage > 50.0:
            print("Mask coverage is too high might overmodify consider raising mask threshold")

        print(f"Starting generation steps {num_inference_steps} guidance {guidance_scale} strength {strength}")
        print(f"Prompt {prompt}")
        if negative_prompt:
            print(f"Negative prompt {negative_prompt}")

        # def make_inpaint_condition(image, image_mask):
        #     image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        #     image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        #     assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        #     image[image_mask > 0.5] = -1.0  # set as masked pixel
        #     image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        #     image = torch.from_numpy(image)
        #     return image

        # fused_image_mask = make_inpaint_condition(rgb_image, mask_image)   

        # def tensor_to_pil_image(tensor):
        #     # Convert tensor to NumPy array and scale it to 0 255
        #     tensor = tensor.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
        #     tensor = np.transpose(tensor, (1, 2, 0))  # Convert CxHxW to HxWxC
        #     tensor = (tensor * 255).astype(np.uint8)  # Scale to 0 255 range and convert to uint8
        #     return Image.fromarray(tensor)
        # fused_image_mask = tensor_to_pil_image(fused_image_mask)

        # mask = np.ones((512, 512), dtype=np.uint8) * 255
        # Image.fromarray(mask, mode="L").save("white_mask_512.png")
        # mask_image = Image.open("white_mask_512.png").convert("L")

        # control_image = Image.new("RGB", (512, 512), (0, 0, 0))

        # control_image = ir_raw.convert("RGB")

        generation_params = {
            "prompt": prompt,
            "image": rgb_image,
            "mask_image": mask_image,
            "control_image": control_image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": torch.Generator(device=self.device).manual_seed(seed) if seed else None,
        }
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt

        result = self.pipe(**generation_params)
        print("Image generation complete")

        if save_debug:
            self.save_debug_images(control_hint, rgb_image, mask_image, control_image)

        return result.images[0], mask_image

    # -------------------------------
    # Debug visualization
    # -------------------------------
    def save_debug_images(self, control_hint, rgb_image, mask_image, control_image, output_dir="debug_output"):
        os.makedirs(output_dir, exist_ok=True)
        control_np = control_hint.permute(1, 2, 0).numpy()

        Image.fromarray((control_np[:, :, 0] * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "01_soft_mask.jpg")
        )
        Image.fromarray((control_np[:, :, 1] * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "02_intensity_map.jpg")
        )
        Image.fromarray((control_np[:, :, 2] * 255).astype(np.uint8), mode="L").save(
            os.path.join(output_dir, "03_gradient_info.jpg")
        )
        rgb_image.save(os.path.join(output_dir, "04_input_rgb.jpg"))
        mask_image.save(os.path.join(output_dir, "05_mask.jpg"))
        control_image.save(os.path.join(output_dir, "06_control_image.jpg"))

        self.save_diagnostic_images(control_hint, output_dir)

        print(f"Debug images saved to {output_dir}")
        print("   - 01_soft_mask.jpg")
        print("   - 02_intensity_map.jpg")
        print("   - 03_gradient_info.jpg")
        print("   - 04_input_rgb.jpg")
        print("   - 05_mask.jpg")
        print("   - 06_control_image.jpg")

    def save_diagnostic_images(self, control_hint, output_dir="debug_output"):
        os.makedirs(output_dir, exist_ok=True)
        control_np = control_hint.permute(1, 2, 0).numpy()
        soft_mask = control_np[:, :, 0]

        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for t in thresholds:
            mask_thresh = (soft_mask > t).astype(np.uint8) * 255
            Image.fromarray(mask_thresh, mode="L").save(
                os.path.join(output_dir, f"07_mask_threshold_{t}.jpg")
            )
        self.save_histogram_analysis(control_hint, output_dir)
        print("   - 07_mask_threshold_*.jpg")
        print("   - 08_histogram_analysis.txt")

    def save_histogram_analysis(self, control_hint, output_dir):
        control_np = control_hint.permute(1, 2, 0).numpy()
        soft_mask = control_np[:, :, 0]
        with open(os.path.join(output_dir, "08_histogram_analysis.txt"), "w") as f:
            f.write("IR image and Mask analysis report\n\n")
            f.write("Soft mask channel 0 statistics\n")
            f.write(f"  Min {soft_mask.min():.6f}\n")
            f.write(f"  Max {soft_mask.max():.6f}\n")
            f.write(f"  Mean {soft_mask.mean():.6f}\n")
            f.write(f"  Median {np.median(soft_mask):.6f}\n")
            f.write(f"  Standard deviation {soft_mask.std():.6f}\n\n")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            f.write("Soft mask quantile analysis\n")
            for p in percentiles:
                val = np.percentile(soft_mask, p)
                f.write(f"  {p:2d} percent {val:.6f}\n")
            f.write("\nCoverage under different thresholds\n")
            for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
                coverage = (soft_mask > t).mean() * 100
                f.write(f"  Threshold {t} {coverage:.2f} percent\n")


# ===============================
# Utility functions
# ===============================

def print_all_parameters(args, generator):
    print("\n" + "=" * 80)
    print("All parameters used for this generation")
    print("=" * 80)

    print("File paths")
    print(f"   Base model {args.base_model}")
    print(f"   RGB image {args.rgb}")
    print(f"   IR image {args.ir}")
    print(f"   Output path {args.output}")

    print("\nGeneration parameters")
    print(f"   Prompt {args.prompt}")
    if args.negative_prompt:
        print(f"   Negative prompt {args.negative_prompt}")
    print(f"   Inference steps {args.steps}")
    print(f"   Guidance scale {args.guidance}")
    print(f"   Generation strength {args.strength}")
    print(f"   Mask threshold {args.mask_threshold}")
    print(f"   Random seed {args.seed if args.seed is not None else 'random'}")

    print(f"\nFlame preset {args.flame_preset}")

    print("\nDebug options")
    print(f"   Save debug images {'yes' if args.debug else 'no'}")

    print("\nDevice info")
    print(f"   Compute device {generator.device}")
    print("   Model precision float16")

    if hasattr(generator, "controlnet") and generator.controlnet:
        total_params = sum(p.numel() for p in generator.controlnet.parameters())
        print(f"   ControlNet parameter count {total_params:,}")

    print("\nReproduction command")
    cmd_parts = [
        "python test.py",
        f"--rgb \"{args.rgb}\"",
        f"--ir \"{args.ir}\"",
        f"--output \"{args.output}\"",
        f"--base-model \"{args.base_model}\"",
        f"--pretrained-controlnet-id \"{args.pretrained_controlnet_id}\"",
        f"--prompt \"{args.prompt}\"",
        f"--steps {args.steps}",
        f"--guidance {args.guidance}",
        f"--strength {args.strength}",
        f"--mask-threshold {args.mask_threshold}",
        f"--flame-preset {args.flame_preset}",
        f"--control-type {args.control_type}",
    ]
    if args.negative_prompt:
        cmd_parts.append(f"--negative-prompt \"{args.negative_prompt}\"")
    if args.seed is not None:
        cmd_parts.append(f"--seed {args.seed}")
    if args.debug:
        cmd_parts.append("--debug")
    if args.lora_path:
        cmd_parts.append(f"--lora-path \"{args.lora_path}\"")
        cmd_parts.append(f"--lora-scale {args.lora_scale}")
    if args.fuse_lora:
        cmd_parts.append("--fuse-lora")

    print("   " + " \\\n     ".join(cmd_parts))
    print("=" * 80)
    print("Parameter print complete")
    print("=" * 80 + "\n")


# ===============================
# Comparison logic
# ===============================

def compare_pretrained_vs_lora(args):
    if args.batch_test_set:
        batch_compare_test_set(args)
    else:
        single_image_compare(args)


def single_image_compare(args):
    print("Starting comparison pretrained ControlNet baseline vs pretrained ControlNet UNet LoRA")

    # Baseline no LoRA
    try:
        pretrained_generator = FlameGenerator(
            base_model_path=args.base_model,
            pretrained_controlnet_id=args.pretrained_controlnet_id,
            lora_path=None,  # Key baseline no LoRA
        )
        print("Pretrained baseline generator no LoRA created successfully")
    except Exception as e:
        print(f"Failed to create pretrained baseline generator {e}")
        return

    # Comparison mount UNet LoRA
    try:
        lora_gen = FlameGenerator(
            base_model_path=args.base_model,
            pretrained_controlnet_id=args.pretrained_controlnet_id,
            lora_path=args.lora_path,  # Key mount only to UNet
            lora_scale=args.lora_scale,
            fuse_lora=args.fuse_lora,
        )
        print("Pretrained comparison generator UNet LoRA created successfully")
    except Exception as e:
        print(f"Failed to create pretrained comparison generator {e}")
        return

    # Generation
    try:
        base_img, _ = pretrained_generator.generate_flame(
            rgb_image_path=args.rgb,
            ir_image_path=args.ir,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength,
            seed=args.seed,
            save_debug=args.debug,
            mask_threshold=args.mask_threshold,
            control_type=args.control_type,
        )
        base_img.save("test_pretrained_baseline.jpg")
        print("Pretrained baseline image test_pretrained_baseline.jpg")
    except Exception as e:
        print(f"Failed to generate pretrained baseline {e}")
        return

    try:
        lora_img, _ = lora_gen.generate_flame(
            rgb_image_path=args.rgb,
            ir_image_path=args.ir,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            strength=args.strength,
            seed=args.seed,
            save_debug=args.debug,
            mask_threshold=args.mask_threshold,
            control_type=args.control_type,
        )
        lora_img.save("test_pretrained_with_lora.jpg")
        print("Pretrained UNet LoRA image test_pretrained_with_lora.jpg")
    except Exception as e:
        print(f"Failed to generate pretrained LoRA {e}")
        return

    print("Comparison complete")


def batch_compare_test_set(args):
    use_lora = hasattr(args, "lora_path") and args.lora_path and os.path.exists(args.lora_path)

    print("=" * 80)
    print("Batch comparison test set pretrained no LoRA vs pretrained UNet LoRA")
    print("=" * 80)

    # Modified part no longer read dataset splits json directly read all data
    from pathlib import Path
    rgb_dir = Path("data/processed_rgb")
    ir_dir = Path("data/thermal")

    if not rgb_dir.exists() or not ir_dir.exists():
        print(f"Input directory does not exist {rgb_dir} or {ir_dir}")
        return

    rgb_images = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
    if not rgb_images:
        print(f"No RGB image files found in {rgb_dir}")
        return

    # Automatically match IR files with the same name
    test_samples = []
    for rgb_path in rgb_images:
        stem = rgb_path.stem
        for ext in [".png", ".jpg", ".jpeg", ".webp",".JPG"]:
            ir_path = ir_dir / f"{stem}{ext}"
            if ir_path.exists():
                test_samples.append({"rgb": rgb_path.name, "ir": ir_path.name})
                break

    if not test_samples:
        print("No matching RGB IR image pairs found please check if file naming is consistent")
        return

    print(f"Test set {len(test_samples)} samples automatically matched from folder")

    # Optional quantity limit
    if args.max_samples:
        test_samples = test_samples[: args.max_samples]
        print(f"   Processing limited to {len(test_samples)} samples")

    # Keep original logic output directory model loading generation loop etc
    output_dir = args.output_dir
    os.makedirs(f"{output_dir}/pretrained", exist_ok=True)
    os.makedirs(f"{output_dir}/lora", exist_ok=True)
    print(f"Output directory {output_dir}")

    pre_dir = os.path.join(output_dir, "pretrained")
    lora_dir = os.path.join(output_dir, "lora")

    existing_pre = []
    existing_lora = []

    # Support jpg jpeg JPG JPEG
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]

    for pattern in patterns:
        existing_pre.extend(glob.glob(os.path.join(pre_dir, pattern)))
        existing_lora.extend(glob.glob(os.path.join(lora_dir, pattern)))

    existing_pre = [os.path.basename(p) for p in existing_pre]
    existing_lora = [os.path.basename(p) for p in existing_lora]

    print("Already exists {} pretrained {} lora files will be automatically skipped".format(
        len(existing_pre), len(existing_lora)
    ))

    # Automatically scan MAX xxx jpg and IRX xxx jpg paired files
    rgb_dir = "./data/processed_rgb"
    ir_dir = "./data/thermal"

    # rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().startswith("max") and f.lower().endswith((".jpg", ".jpeg", ".png"))])
    # ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().startswith("irx") and f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # print(f"RGB file count {len(rgb_files)} example {rgb_files[:5]}")
    # print(f"IRX file count {len(ir_files)} example {ir_files[:5]}")


    # # Align by number in filename for example MAX_0001.jpg IRX_0001.jpg
    # pairs = []
    # for rgb in rgb_files:
    #     idx = rgb.lower().replace("max_", "").replace(".jpg", "")
    #     for ir in ir_files:
    #         if idx == ir.lower().replace("irx_", "").replace(".jpg", ""):
    #             pairs.append({"rgb": rgb, "ir": ir})
    #             break

    # test_samples = pairs
    # print(f"Matched {len(test_samples)} pairs of MAX IRX images")

    print("\nInitializing generator")
    # Pretrained branch no LoRA
    try:
        pretrained_generator = FlameGenerator(
            base_model_path=args.base_model,
            pretrained_controlnet_id=args.pretrained_controlnet_id,
            lora_path=None,  # No LoRA
        )
        print("Pretrained generator no LoRA")
    except Exception as e:
        print(f"Pretrained generator failed {e}")
        return

    # Pretrained UNet LoRA
    try:
        lora_generator = FlameGenerator(
            base_model_path=args.base_model,
            pretrained_controlnet_id=args.pretrained_controlnet_id,
            lora_path=(args.lora_path if use_lora else None),
            lora_scale=args.lora_scale,
            fuse_lora=args.fuse_lora,
        )
        print("Pretrained generator " + ("with UNet LoRA" if use_lora else "no LoRA lora path not provided"))
    except Exception as e:
        print(f"Generator failed {e}")
        return

    print(f"\nStarting batch generation {len(test_samples)} samples")
    print("=" * 80)

    start_time = time.time()
    generated_count = 0  # Record actual generated count excluding skipped


    success_count, failed_count, skipped = 0, 0, 0
    for idx, sample in enumerate(tqdm(test_samples, desc="Generating", ncols=80)):
        rgb_filename = sample["rgb"]
        ir_filename = sample["ir"]
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        ir_path = os.path.join(ir_dir, ir_filename)

        # Consistent filenames used to filter already generated
        save_name = f"{rgb_filename}"
        if save_name in existing_pre or save_name in existing_lora:
            skipped += 1
            continue

        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            failed_count += 1
            continue

        try:
            sample_seed = args.seed if args.seed is not None else (42 + idx)

            # Pretrained no LoRA
            # base_img, _ = pretrained_generator.generate_flame(
            #     rgb_image_path=rgb_path,
            #     ir_image_path=ir_path,
            #     prompt=args.prompt,
            #     negative_prompt=args.negative_prompt,
            #     num_inference_steps=args.steps,
            #     guidance_scale=args.guidance,
            #     strength=args.strength,
            #     seed=sample_seed,
            #     save_debug=False,
            #     mask_threshold=args.mask_threshold,
            #     control_type=args.control_type,
            # )
            # base_img.save(os.path.join(pre_dir, save_name))

            # Pretrained UNet LoRA
            lora_img, _ = lora_generator.generate_flame(
                rgb_image_path=rgb_path,
                ir_image_path=ir_path,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                strength=args.strength,
                seed=sample_seed,
                save_debug=False,
                mask_threshold=args.mask_threshold,
                control_type=args.control_type,
            )
            lora_img.save(os.path.join(lora_dir, save_name))

            success_count += 1
            generated_count += 1

        except Exception as e:
            print(f"\nSample {idx:04d} failed {e}")
            failed_count += 1
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate average time avoid division by zero
    if generated_count > 0:
        avg_time = total_time / generated_count
    else:
        avg_time = 0.0

    

    print("\n" + "=" * 80)
    print("Batch generation complete")
    print("=" * 80)
    print("Statistics")
    print(f"   Total samples {len(test_samples)}")
    print(f"   Success {success_count}")
    print(f"   Failed {failed_count}")
    print(f"   Skipped {skipped}")
    print("\nResults saved in")
    print(f"   {output_dir}/pretrained/ pretrained no LoRA")
    print(f"   {output_dir}/lora/ pretrained with UNet LoRA")
    print("=" * 80)


    print("-" * 40)
    print("Time statistics")
    print(f"   Total time {total_time:.2f} seconds {total_time/60:.2f} minutes")
    if generated_count > 0:
        print(f"   Actual generated samples {generated_count}")
        print(f"   Average time per sample {avg_time:.4f} seconds per image")
        # If one loop generates 2 images Pretrained LoRA can add explanation
        # print(f"   Note each sample contains 2 generations single generation is about {avg_time/2:.4f} seconds")
    else:
        print("   No actual generation tasks executed all skipped or failed")
        
    print("-" * 40)


# ===============================
# main
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Generate flame images using pretrained ControlNet optional UNet LoRA")
    parser.add_argument("--rgb", type=str, required=False, help="RGB background image path required for single generation")
    parser.add_argument("--ir", type=str, required=False, help="IR intensity image path required for single generation")
    parser.add_argument("--output", type=str, default="./generated_flame.jpg", help="Output image path")
    parser.add_argument("--base-model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model path or ID")
    parser.add_argument("--pretrained-controlnet-id", type=str, default="lllyasviel/control_v11p_sd15_inpaint",
                        help="Pretrained ControlNet model ID")

    parser.add_argument("--prompt", type=str,
                        default="A vibrant, bright, intense wildfire with glowing orange and yellow flames, high-resolution, dramatic lighting, realistic fire textures",
                        help="Text prompt")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=10, help="Guidance scale")
    parser.add_argument("--strength", type=float, default=1.0, help="Generation strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--mask-threshold", type=float, default=0.4, help="Mask threshold 0.0 to 1.0")
    parser.add_argument("--flame-preset", type=str, default="normal",
                        choices=["normal", "bright", "intense", "warm", "no-blue"], help="Flame preset")
    parser.add_argument("--negative-prompt", type=str, default="", help="Custom negative prompt")
    parser.add_argument("--reference-dataset", type=str, default="./wildfire_images/FLAME_sample",
                        help="Reference dataset path for FID calculation currently disabled")
    parser.add_argument("--control-type", type=str, default="rgb", choices=["rgb", "fused", "single"], help="Control image type")

    parser.add_argument("--compare-pretrained", action="store_true", help="Compare pretrained no LoRA vs pretrained UNet LoRA")
    parser.add_argument("--batch-test-set", action="store_true", help="Batch process all samples in the test set")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit the number of samples processed")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Batch comparison results output directory")

    # UNet LoRA injected via pipeline.load_lora_weights into UNet
    parser.add_argument("--lora-path", type=str, default=None, help="UNet LoRA weight path safetensors or directory")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale")
    parser.add_argument("--fuse-lora", action="store_true", help="Whether to fuse LoRA into UNet")

    args = parser.parse_args()

    # Color presets
    print(f"Applying flame preset {args.flame_preset}")
    if args.flame_preset == "no-blue":
        args.prompt = "Realistic wildfire with bright orange and red flames, warm fire colors, yellow flames, hot glowing fire, natural fire colors, orange glow, red hot flames"
        args.guidance = max(args.guidance, 10.0)
        args.strength = max(args.strength, 0.8)
        if not args.negative_prompt:
            args.negative_prompt = "blue, cyan, turquoise, purple, violet, green, cold colors, sky, water, ice, snow, blue flames, blue fire, cold fire"
        print("   Avoid blue preset activated")
    elif args.flame_preset == "warm":
        args.prompt = "Warm wildfire with orange, red and yellow flames, hot colors, warm lighting, golden flames, amber fire, bright warm fire"
        args.guidance = max(args.guidance, 9.0)
        args.strength = max(args.strength, 0.8)
        if not args.negative_prompt:
            args.negative_prompt = "blue, cyan, purple, green, cold colors, cool tones, sky, water, ice"
        print("   Warm tone preset activated")
    elif args.flame_preset == "bright":
        args.prompt = "Bright, vibrant wildfire with intense orange and yellow colors, red flames, dramatic lighting, glowing fire, bright flames, luminous fire"
        args.guidance = max(args.guidance, 10.0)
        args.strength = max(args.strength, 0.8)
        if not args.negative_prompt:
            args.negative_prompt = "blue, cyan, purple, green, cold colors, sky, water, ash, ember, dark, dim, faded"
        print("   Bright flame preset activated")
    elif args.flame_preset == "intense":
        args.prompt = "Intense roaring wildfire with brilliant orange, red and yellow flames, hot fire colors, powerful flames, blazing fire, fierce flames"
        args.guidance = max(args.guidance, 12.0)
        args.strength = max(args.strength, 0.9)
        args.steps = max(args.steps, 25)
        if not args.negative_prompt:
            args.negative_prompt = "blue, cyan, purple, green, cold colors, sky, water, ice, weak flames, dim fire"
        print("   Intense flame preset activated")
    else:
        if not args.negative_prompt:
            args.negative_prompt = "blue, cyan, purple, green, cold colors, sky, water,violet color"
        print("   Normal flame preset activated")

    # Mode dispatch
    need_single_inputs = not (args.compare_pretrained or args.batch_test_set)
    if need_single_inputs:
        if not args.rgb or not args.ir:
            print("Single generation mode requires rgb and ir parameters")
            return
        if not os.path.exists(args.rgb):
            print(f"RGB image does not exist {args.rgb}")
            return
        if not os.path.exists(args.ir):
            print(f"IR image does not exist {args.ir}")
            return

    print("Starting to generate flame images")
    print(f"Using prompt {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt {args.negative_prompt}")

    # Compare or Batch Compare
    if args.compare_pretrained or args.batch_test_set:
        print("Comparison mode pretrained no LoRA vs pretrained UNet LoRA")
        compare_pretrained_vs_lora(args)
        return

    # Single generation path decides whether to mount LoRA based on parameters
    generator = FlameGenerator(
        base_model_path=args.base_model,
        pretrained_controlnet_id=args.pretrained_controlnet_id,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        fuse_lora=args.fuse_lora,
    )

    gen_img, _ = generator.generate_flame(
        rgb_image_path=args.rgb,
        ir_image_path=args.ir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        strength=args.strength,
        seed=args.seed,
        save_debug=args.debug,
        mask_threshold=args.mask_threshold,
        control_type=args.control_type,
    )

    gen_img.save(args.output)
    print(f"Flame image saved to {args.output}")
    print_all_parameters(args, generator)


if __name__ == "__main__":
    main()