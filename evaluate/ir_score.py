import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# Configuration
# Two generated image folders to compare
FAKE_DIR_LORA = "./comparison_results_v4_10_1_rank16_all_nsoftmask_yellow/lora"     # LoRA model generated image path
FAKE_DIR_BASE = "./images_inpaint_sd/pretrained"                     # Base model generated image path
# FAKE_DIR_BASE = "./comparison_results_v4_10_1_rank16_all_sd15/lora/"                     # Base model generated image path


# Corresponding IR images and mask folders
IR_DIR = "/home/cx/diffusion/FLAME_SD-main/data/thermal"           # Original IR image path
MASK_DIR = "./mask_info_v2"                                        # Mask image path

# Image resolution used for calculation
IMAGE_SIZE = 512

# IR similarity calculation function
def calculate_adherence_to_ir(generated_image_paths, ir_image_paths, mask_image_paths):
    """
    Calculate the average similarity metrics between a set of generated images and corresponding IR images
    """
    all_pearson, all_ssim, all_rmse = [], [], []

    # Use tqdm to display progress bar
    iterator = tqdm(zip(generated_image_paths, ir_image_paths, mask_image_paths), 
                    total=len(generated_image_paths), 
                    desc=f"Calculating IR Adherence for {os.path.basename(os.path.dirname(generated_image_paths[0]))}")

    for gen_path, ir_path, mask_path in iterator:
        try:
            # 1 Load and prepare data
            generated_img_pil = Image.open(gen_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            ir_img_pil = Image.open(ir_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            mask_img_pil = Image.open(mask_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))

            mask_np = np.array(mask_img_pil)
            bool_mask = mask_np > 128

            # Extract perceived intensity from generated image
            generated_gray = np.array(generated_img_pil.convert("L"))
            generated_intensity = np.zeros_like(generated_gray, dtype=np.float32)
            generated_intensity[bool_mask] = generated_gray[bool_mask]

            # Extract physical intensity from original IR image
            ir_np = np.array(ir_img_pil)
            ir_intensity_map = np.zeros_like(ir_np, dtype=np.float32)
            ir_intensity_map[bool_mask] = ir_np[bool_mask]

            # Extract values within the mask region
            gen_values = generated_intensity[bool_mask]
            ir_values = ir_intensity_map[bool_mask]

            if gen_values.size < 2: continue

            # 2 Calculate metrics
            pearson_corr = np.corrcoef(gen_values, ir_values)[0, 1]
            if not np.isnan(pearson_corr): all_pearson.append(pearson_corr)
            
            ssim_score = ssim(generated_intensity, ir_intensity_map, data_range=255.0)
            all_ssim.append(ssim_score)
            
            def normalize(arr):
                ptp = np.ptp(arr)
                return (arr - arr.min()) / ptp if ptp > 0 else arr
            
            rmse_normalized = np.sqrt(np.mean((normalize(gen_values) - normalize(ir_values)) ** 2))
            all_rmse.append(rmse_normalized)

        except Exception as e:
            print(f"\nSkipping {os.path.basename(gen_path)} due to error {e}")

    return {
        "avg_pearson": np.mean(all_pearson) if all_pearson else 0,
        "avg_ssim": np.mean(all_ssim) if all_ssim else 0,
        "avg_rmse_norm": np.mean(all_rmse) if all_rmse else 0
    }

# Main process
def main():
    print("Start calculating similarity between generated images and IR images")

    # 1 Scan and align file paths
    print("\nStep 1 of 3 Scanning and aligning file paths")
    
    # Align based on files in the LoRA folder
    lora_paths = sorted([os.path.join(FAKE_DIR_LORA, f) for f in os.listdir(FAKE_DIR_LORA) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def align_paths(ref_paths, target_dir):
        ref_stems = {Path(p).stem: p for p in ref_paths}
        aligned = []
        target_files = {Path(f).stem: os.path.join(target_dir, f) for f in os.listdir(target_dir)}
        
        for stem, ref_path in ref_stems.items():
            if stem in target_files:
                aligned.append(target_files[stem])
            else:
                print(f"Warning Could not find matching file for {os.path.basename(ref_path)} in {target_dir} will skip this sample")
        return aligned
    
    base_paths = align_paths(lora_paths, FAKE_DIR_BASE)
    ir_paths = align_paths(lora_paths, IR_DIR)
    mask_paths = align_paths(lora_paths, MASK_DIR)
    
    # Ensure all lists have the same length
    min_len = min(len(lora_paths), len(base_paths), len(ir_paths), len(mask_paths))
    lora_paths = lora_paths[:min_len]
    base_paths = base_paths[:min_len]
    ir_paths = ir_paths[:min_len]
    mask_paths = mask_paths[:min_len]

    print(f"File alignment complete found {min_len} matched sample pairs")

    # 2 Calculate metrics for LoRA model
    print("\nStep 2 of 3 Calculating similarity metrics for LoRA model")
    ir_adherence_lora = calculate_adherence_to_ir(lora_paths, ir_paths, mask_paths)
    
    # 3 Calculate metrics for base model
    print("\nStep 3 of 3 Calculating similarity metrics for base model")
    ir_adherence_base = calculate_adherence_to_ir(base_paths, ir_paths, mask_paths)
    
    # Print final results
    print("\nFinal Evaluation Results")
    print("\nThis evaluation measures how well the generated flame follows the structure of the original IR image")
    
    print(f"\nLoRA Model {os.path.basename(FAKE_DIR_LORA)}")
    print(f"  IR Pearson Correlation Coefficient {ir_adherence_lora['avg_pearson']:.4f} higher is better closer to 1")
    print(f"  IR Structural Similarity SSIM {ir_adherence_lora['avg_ssim']:.4f} higher is better closer to 1")
    print(f"  IR Normalized RMSE {ir_adherence_lora['avg_rmse_norm']:.4f} lower is better closer to 0")
    
    print(f"\nBase Model {os.path.basename(FAKE_DIR_BASE)}")
    print(f"  IR Pearson Correlation Coefficient {ir_adherence_base['avg_pearson']:.4f} higher is better closer to 1")
    print(f"  IR Structural Similarity SSIM {ir_adherence_base['avg_ssim']:.4f} higher is better closer to 1")
    print(f"  IR Normalized RMSE {ir_adherence_base['avg_rmse_norm']:.4f} lower is better closer to 0")
    
    print("\n")
    print("Script execution finished")

if __name__ == "__main__":
    main()