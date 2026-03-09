import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import cv2  # Added import

# Hugging Face Transformers import
from transformers import CLIPProcessor, CLIPModel

# Configuration
# REAL_DIR = "./evaluate/real_images"       # Real image path
REAL_DIR = "/home/cx/diffusion/FLAME_SD-main/data/processed_rgb"       # Real image path

FAKE_DIR = "./comparison_results_v4_10_1_rank16_all_nsoftmask_yellow/lora"  # Generated image path
# FAKE_DIR_base = "./images_scribble_sd_v3/pretrained/"  # Generated image path
FAKE_DIR_base = "./images_inpaint_sd/pretrained"  # Generated image path


MASK_DIR = "./mask_info_v2"                  # Mask image path
BATCH_SIZE = 32
IMAGE_SIZE = 299
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Preprocessing FID
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load Images modified
def load_images(folder, max_images=None, transform=None):
    """Load images as Tensors and return file paths"""
    paths = []
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG'):
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
    
    print(f"Found {len(paths)} image files in {folder}")
    
    paths = sorted(paths)
    for p in paths[:5]:
        print(os.path.basename(p))
    print("")
    if max_images:
        paths = paths[:max_images]
    
    imgs = []
    for p in tqdm(paths, desc=f"Loading {folder}"):
        with Image.open(p) as img:
            imgs.append(transform(img.convert("RGB")) if transform else transform(img))
    
    tensor_data = torch.stack(imgs) if imgs else None
    return tensor_data, paths  # Return data and paths

def load_mask(folder, max_images=None):
    """Load mask images as Tensors and return file paths"""
    paths = []
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG'):
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
        
    paths = sorted(paths)
    for p in paths[:5]:
        print(os.path.basename(p))
    print("")
    if max_images:
        paths = paths[:max_images]
        
    masks = []
    for p in tqdm(paths, desc=f"Loading masks"):
        try:
            with Image.open(p) as mask:
                masks.append(mask_transform(mask.convert("RGB")))
        except Exception as e:
            print(f"Skipping unreadable mask {p} {e}")
            
    tensor_data = torch.stack(masks) if masks else None
    return tensor_data, paths # Return data and paths

# Added CLIP Score and Confidence Calculation
# Added CLIP Score and Confidence Calculation using cosine similarity
def calculate_clip_metrics(image_paths, mask_paths, prompt, classes, clip_model, clip_processor):
    """
    Calculate CLIP Score and CLIP Confidence using Hugging Face CLIP model
    CLIP Score Similarity between the whole image and prompt based on cosine similarity
    CLIP Confidence Confidence score of the mask region being classified as the target category
    """
    clip_model.to(DEVICE)
    clip_model.eval()
    
    all_clip_scores = []
    all_confidences = []

    with torch.no_grad():
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc="Calculating CLIP Metrics"):
            try:
                image = Image.open(img_path).convert("RGB")
                # mask = Image.open(mask_path).convert("L") # Load as grayscale image

                # image_np = np.array(image)
                # mask_np = np.array(mask)
                
                # # Normalize mask to 0 1 and expand to 3 channels
                # mask_normalized = mask_np / 255.0
                # mask_3d = np.stack([mask_normalized] * 3, axis=-1)
                
                # # Apply mask background becomes black
                # masked_image_np = (image_np * mask_3d).astype(np.uint8)
                # masked_image_pil = Image.fromarray(masked_image_np)
                
                # CLIP Score cosine similarity START
                # 1 Prepare image and text inputs using processor respectively
                inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)
                
                # 2 Get image and text feature vectors respectively these features are already normalized
                image_features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                text_features = clip_model.get_text_features(input_ids=inputs['input_ids'])
                
                # 3 Calculate cosine similarity
                # Since features are already normalized dot product equals cosine similarity but using library function is more explicit
                cosine_sim = torch.nn.functional.cosine_similarity(image_features, text_features)
                clip_score = cosine_sim.item()
                similarities = (image_features @ text_features.T).diag().cpu().numpy()
                all_clip_scores.append(similarities)
                # CLIP Score cosine similarity END

                # CLIP Confidence keep unchanged START
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours: 
                    continue
                
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                if w > 10 and h > 10:
                    cropped_image = image.crop((x, y, x + w, y + h))
                    
                    # This logic remains unchanged as it is a classification task
                    inputs_conf = clip_processor(text=classes, images=cropped_image, return_tensors="pt", padding=True).to(DEVICE)
                    outputs_conf = clip_model(**inputs_conf)
                    probs = outputs_conf.logits_per_image.softmax(dim=1)
                    confidence = probs[0, 0].item()
                    all_confidences.append(confidence)
                # CLIP Confidence keep unchanged END

            except Exception as e:
                print(f"Skipping file due to error {img_path} {e}")

    avg_clip_score = np.mean(all_clip_scores) if all_clip_scores else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    return avg_clip_score, avg_confidence
# FID Related Functions keep unchanged
def apply_mask(images, masks):
    if images is None or masks is None: raise ValueError("Images or masks are None")
    return images * masks

def get_inception_features(images, model):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting features"):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            pred = model(batch)
            feats.append(pred.detach().cpu())
    return torch.cat(feats, dim=0).numpy()

def calculate_mask_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

# Main Process modified
def main():
    print("Script started")
    print(f"Using device {DEVICE}")

    # 1 Load InceptionV3 for FID
    print("\nStep 1 of 7 Loading InceptionV3 model")
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.to(DEVICE)
    print("InceptionV3 model loaded successfully")
    
    # 2 Load CLIP model for CLIP Score
    print("\nStep 2 of 7 Loading CLIP model")
    # Modified code
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", 
        variant="fp16",              # Specify loading fp16 version weights
        use_safetensors=True         # Force using safetensors file
    ).to(DEVICE)

    # Loading of processor does not need to change
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully")

    # 3 Load images and masks
    print("\nStep 3 of 7 Loading images and masks")
    real_imgs, _ = load_images(REAL_DIR, transform=transform)
    fake_imgs, fake_paths = load_images(FAKE_DIR, max_images=1193, transform=transform)
    fake_imgs_base, fake_paths_base = load_images(FAKE_DIR_base, max_images=1193, transform=transform)
    masks, mask_paths = load_mask(MASK_DIR,max_images=1193)

    # error checking part keeps unchanged
    if any(x is None for x in [real_imgs, fake_imgs, fake_imgs_base, masks]):
        print("Error a dataset failed to load script terminated")
        return
    
    print(f"\nData loaded successfully {len(real_imgs)} real images {len(fake_imgs)} generated images {len(masks)} masks")
    
    # 4 Calculate CLIP Score and Confidence
    print("\nStep 4 of 7 Calculating CLIP scores")
    prompt = "wildfire image"
    classes = ["wildfire", "non-fire"]
    
    clip_score, clip_conf = calculate_clip_metrics(fake_paths, mask_paths, prompt, classes, clip_model, clip_processor)
    clip_score_base, clip_conf_base = calculate_clip_metrics(fake_paths_base, mask_paths, prompt, classes, clip_model, clip_processor)

    # 5 Apply mask for FID
    print("\nStep 5 of 7 Applying masks")
    # masked_fake_imgs = apply_mask(fake_imgs, masks)
    # masked_fake_imgs_base = apply_mask(fake_imgs_base, masks)
    print("Masks applied successfully")

    # 6 Extract features for FID
    print("\nStep 6 of 7 Extracting features")
    real_feats = get_inception_features(real_imgs, inception)
    fake_feats = get_inception_features(fake_imgs, inception)
    fake_feats_base = get_inception_features(fake_imgs_base, inception)
    print("Feature extraction completed")

    # 7 Calculate mean covariance and FID
    print("\nStep 7 of 7 Calculating FID scores")
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    mu3, sigma3 = np.mean(fake_feats_base, axis=0), np.cov(fake_feats_base, rowvar=False)
    
    mask_fid = calculate_mask_fid(mu1, sigma1, mu2, sigma2)
    mask_fid_base = calculate_mask_fid(mu1, sigma1, mu3, sigma3)
    
    # Final result printing
    print("\nFinal Evaluation Results")
    print(f"\nLoRA Model {FAKE_DIR}")
    print(f"  Mask FID Score {round(mask_fid, 4)}")
    print(f"  CLIP Score     {round(clip_score, 4)}")
    print(f"  CLIP Confidence {round(clip_conf, 4)}")
    
    print(f"\nBase Model {FAKE_DIR_base}")
    print(f"  Mask FID Score {round(mask_fid_base, 4)}")
    print(f"  CLIP Score     {round(clip_score_base, 4)}")
    print(f"  CLIP Confidence {round(clip_conf_base, 4)}")
    print("\n")
    print("Script execution finished")

if __name__ == "__main__":
    main()