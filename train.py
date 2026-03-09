#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2

from accelerate import Accelerator
from accelerate.utils import set_seed
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from safetensors.torch import save_file


# ---------------------------
# Dataset
# ---------------------------
class PairFolder(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, tokenizer, size=512, center_crop=False, list_limit: int = 0):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"[data] not found: {self.root}")

        all_files = sorted([p for p in self.root.iterdir() if p.is_file()])
        self.imgs = [p for p in all_files if p.suffix.lower() in self.IMG_EXTS]
        if list_limit and len(self.imgs) > list_limit:
            self.imgs = self.imgs[:list_limit]

        if len(self.imgs) == 0:
            raise ValueError(f"[data] no images in {self.root}")

        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

       
        self.tfm = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(size) if center_crop else transforms.RandomResizedCrop(size, scale=(0.9, 1.0))),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
        ])

        # check captions
        self.missing_txt = []
        for p in self.imgs:
            if not p.with_suffix(".txt").exists():
                self.missing_txt.append(p.name)
        if self.missing_txt:
            msg = f"[data] {len(self.missing_txt)} images missing .txt captions. First 10: {self.missing_txt[:10]}"
            raise ValueError(msg)

   
    @staticmethod
    def _neutralize_yellow(pil_img: Image.Image):
        np_img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # PIL->BGR
        lab = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        b_mean = np.mean(b)
        b_adj = b.astype(np.float32) - (b_mean - 128) * 0.2  
        b_adj = np.clip(b_adj, 0, 255).astype(np.uint8)
        lab_fixed = cv2.merge([L, a, b_adj])
        fixed = cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2BGR)
        fixed = fixed[:, :, ::-1]  # back to RGB
        return Image.fromarray(fixed)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_path = self.imgs[i]
        txt_path = img_path.with_suffix(".txt")

        image = Image.open(img_path).convert("RGB")
       
        image = self._neutralize_yellow(image)
        pixel_values = self.tfm(image)

        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        ids = self.tokenizer(
            caption, truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": ids.squeeze(0)}


# ---------------------------
# Save LoRA helpers
# ---------------------------
def save_lora_from_unet(accelerator: Accelerator, unet, out_dir: str, filename: str):
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(unet)
    try:
        from peft.utils import get_peft_model_state_dict
        state = get_peft_model_state_dict(unwrapped)
        os.makedirs(out_dir, exist_ok=True)
        safepath = os.path.join(out_dir, filename)
        save_file(state, safepath)
        accelerator.print(f"[save] LoRA -> {safepath}")
    except Exception as e:
        accelerator.print(f"[save failed] {e}")





# ---------------------------
# Eval loop (optional)
# ---------------------------
@torch.no_grad()
def eval_on_loader(vae, unet, text_encoder, noise_scheduler, dl, accelerator, max_batches=2):
    unet.eval()
    losses = []
    it = 0
    for batch in dl:
        if it >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)

        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.size(0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = text_encoder(input_ids)[0]

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        # gather for multi-process average
        losses.append(accelerator.gather_for_metrics(loss.detach().float()).mean().item())
        it += 1

    unet.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------
# Train
# ---------------------------
def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.tb_dir else None,
        project_dir=args.tb_dir if args.tb_dir else None,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Env / versions
    accelerator.print("=" * 80)
    accelerator.print("[env] torch", torch.__version__)
    accelerator.print("[env] diffusers", diffusers.__version__)
    import accelerate as _acc
    accelerator.print("[env] accelerate", _acc.__version__)
    accelerator.print("[env] device", accelerator.device)
    accelerator.print("=" * 80)

    # ---- Load base
    base = args.pretrained_model_name_or_path  # "runwayml/stable-diffusion-v1-5"
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # xformers / grad checkpointing
    if args.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            accelerator.print("[opt] xFormers attention enabled")
        except Exception as e:
            accelerator.print(f"[opt] xFormers enable failed: {e}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        accelerator.print("[opt] gradient checkpointing enabled")

    # Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # LoRA inject
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    unet.add_adapter(lora_cfg)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    accelerator.print(f"[model] trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Data
    dataset = PairFolder(args.instance_data_dir, tokenizer, size=args.resolution,
                         center_crop=args.center_crop, list_limit=args.debug_limit)
    accelerator.print(f"[data] train images: {len(dataset)}  (dir: {args.instance_data_dir})")
    accelerator.print(f"[data] sample: {dataset.imgs[0].name} (caption={dataset.imgs[0].with_suffix('.txt').name})")

    def collate_fn(examples):
        px = torch.stack([ex["pixel_values"] for ex in examples], dim=0).to(memory_format=torch.contiguous_format).float()
        ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)
        return {"pixel_values": px, "input_ids": ids}

    dl = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Optional validation
    val_dl = None
    if args.val_data_dir:
        try:
            val_set = PairFolder(args.val_data_dir, tokenizer, size=args.resolution,
                                 center_crop=True, list_limit=args.val_limit)
            val_dl = DataLoader(val_set, batch_size=min(4, args.train_batch_size),
                                shuffle=False, num_workers=max(1, args.num_workers//2),
                                collate_fn=collate_fn, pin_memory=True)
            accelerator.print(f"[val] enabled: {len(val_set)} images (dir: {args.val_data_dir})")
        except Exception as e:
            accelerator.print(f"[val] disabled (load failed): {e}")

    # Optim / LR
    optimizer = AdamW(filter(lambda p: p.requires_grad, unet.parameters()),
                      lr=args.learning_rate, weight_decay=args.adam_weight_decay)

    total_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, num_training_steps=total_steps
    )

    unet, optimizer, dl, lr_scheduler = accelerator.prepare(unet, optimizer, dl, lr_scheduler)

    # Devices
    vae.to(accelerator.device, dtype=torch.float16)
    # text encoder fp32 is often more stable
    text_encoder.to(accelerator.device, dtype=torch.float32)

    # TQDM
    from tqdm.auto import tqdm
    pbar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process, desc="train")

    # TensorBoard
    if accelerator.log_with == "tensorboard":
        accelerator.init_trackers("lora_unet")

    # ---- Best trackers (init before loop)
    best_loss = float("inf")
    best_step = -1

    # Train Loop
    step = 0
    unet.train()
    last_log_t = time.time()

    try:
        while step < total_steps:
            for batch in dl:
                if step >= total_steps:
                    break

                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)

                    # latents
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # noise & timestep
                    noise = torch.randn_like(latents)
                    bsz = latents.size(0)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                              (bsz,), device=latents.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # text
                    input_ids = batch["input_ids"].to(accelerator.device)
                    encoder_hidden_states = text_encoder(input_ids)[0]

                    # predict noise
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    # ---- Best model tracking (avg across processes)
                    loss_val = accelerator.gather_for_metrics(loss.detach().float()).mean().item()
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_step = step
                        save_lora_from_unet(accelerator, unet, args.output_dir, "best_lora.safetensors")
                        if accelerator.is_main_process:
                            accelerator.print(f"[best] step={step} loss={best_loss:.6f}")

                    accelerator.backward(loss)

                    if args.max_grad_norm and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                step += 1
                pbar.update(1)

                # logs
                if step % args.log_every == 0 or (time.time() - last_log_t) > 30:
                    lr = lr_scheduler.get_last_lr()[0]
                    loss_disp = accelerator.gather_for_metrics(loss.detach().float()).mean().item()
                    pbar.set_postfix({"loss": f"{loss_disp:.4f}", "lr": f"{lr:.2e}"})
                    if accelerator.is_main_process:
                        print(f"[step {step:>6}] loss={loss_disp:.6f} lr={lr:.3e}")
                    if accelerator.log_with == "tensorboard":
                        accelerator.log({"train/loss": loss_disp, "train/lr": lr}, step=step)
                    last_log_t = time.time()

                # eval
                if val_dl is not None and args.eval_every > 0 and step % args.eval_every == 0:
                    val_loss = eval_on_loader(vae, unet, text_encoder, noise_scheduler, val_dl, accelerator, max_batches=args.val_batches)
                    if accelerator.is_main_process:
                        print(f"[eval @ {step}] val_loss={val_loss:.6f}")
                    if accelerator.log_with == "tensorboard":
                        accelerator.log({"val/loss": val_loss}, step=step)

                # save (periodic)
                if args.save_every > 0 and step % args.save_every == 0:
                    save_lora_from_unet(accelerator, unet, args.output_dir, f"lora_step_{step}.safetensors")

        # final save
        save_lora_from_unet(accelerator, unet, args.output_dir, "final_lora.safetensors")
        if accelerator.is_main_process:
            print(f"[done] total_steps={total_steps}")
            print(f"Best model: step={best_step}, loss={best_loss:.6f} (saved as best_lora.safetensors)")
            # write summary
            with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(
                    f"best_step: {best_step}\n"
                    f"best_loss: {best_loss:.6f}\n"
                    f"total_steps: {total_steps}\n"
                    f"batch_size: {args.train_batch_size}\n"
                    f"lr: {args.learning_rate}\n"
                    f"rank/alpha: {args.lora_rank}/{args.lora_alpha}\n"
                    f"dataset: {args.instance_data_dir}\n"
                )

    except KeyboardInterrupt:
        save_lora_from_unet(accelerator, unet, args.output_dir, f"interrupt_step_{step}.safetensors")
        raise
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n[error] {e}\nSaving emergency checkpoint...")
        save_lora_from_unet(accelerator, unet, args.output_dir, f"error_step_{step}.safetensors")
        raise
    finally:
        pbar.close()
        accelerator.end_training()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train UNet LoRA (SD1.5) with detailed logging")

    # base & data
    ap.add_argument("--pretrained_model_name_or_path", type=str,
                    default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--instance_data_dir", type=str, required=True)
    ap.add_argument("--val_data_dir", type=str, default=None, help="optional small validation folder")
    ap.add_argument("--val_limit", type=int, default=0, help="subset validation images (0 = all)")
    ap.add_argument("--output_dir", type=str, default="lora_output")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--center_crop", action="store_true")
    ap.add_argument("--debug_limit", type=int, default=0, help="subset train images for quick dry-run")

    # train
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--max_train_steps", type=int, default=6000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=0, help="set >0 to enable eval every N steps")
    ap.add_argument("--val_batches", type=int, default=2, help="eval on how many mini-batches")

    # optim & sched
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--adam_weight_decay", type=float, default=1e-2)
    ap.add_argument("--lr_scheduler", type=str, default="cosine")
    ap.add_argument("--lr_warmup_steps", type=int, default=100)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # system
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--enable_xformers", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--tb_dir", type=str, default=None, help="tensorboard log dir (enables TB logging)")
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)