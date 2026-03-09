import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from scipy import linalg

# Configuration
REAL_DIR = "./data/fire_pairs_new"       # Real image path
# REAL_DIR = "/home/cx/diffusion/FLAME_SD-main/data/processed_rgb"
FAKE_DIR = "./comparison_results_v3_10_1_rank16_all/lora"  # Generated image path
FAKE_DIR_base = "./images_scribble/pretrained"  # Generated image path

MASK_DIR = "./mask_info"                  # Mask image path
BATCH_SIZE = 32
IMAGE_SIZE = 299
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Preprocessing
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

# Load Images
def load_images(folder, max_images=None, transform=None):
    """Load images as Tensors"""
    paths = []
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG'):
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
    
    # Output file list
    print(f"Found {len(paths)} image files in {folder}")
    
    paths = sorted(paths)
    for p in paths[:5]: # Only print the first 5 to avoid spamming
        print(os.path.basename(p)) # os.path.basename only shows the file name clearer
    print("")
    if max_images:
        paths = paths[:max_images]
    imgs = []
    for p in tqdm(paths, desc=f"Loading {folder}"):
        
        with Image.open(p) as img:
            imgs.append(transform(img.convert("RGB")) if transform else transform(img))
        
    
    if not imgs:
        print("No images loaded")
    
    return torch.stack(imgs) if imgs else None

def load_mask(folder, max_images=None):
    """Load mask images as Tensors"""
    paths = []
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG'):
        paths.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(ext)])
    paths = sorted(paths)

    for p in paths[:5]: # Only print the first 5 to avoid spamming
        print(os.path.basename(p)) # os.path.basename only shows the file name clearer
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
    
    return torch.stack(masks) if masks else None

# Apply Mask
def apply_mask(images, masks):
    """Apply mask to images"""
    if images is None or masks is None:
        raise ValueError("Images or masks are None cannot apply mask")
    return images * masks

# Extract Features
def get_inception_features(images, model):
    """Extract features using InceptionV3 2048 dims"""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting features"):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            pred = model(batch)
            feats.append(pred.detach().cpu())
    return torch.cat(feats, dim=0).numpy()

# Calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Frechet Inception Distance"""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

# Calculate Masked FID
def calculate_mask_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Masked FID"""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    mask_fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(mask_fid)

# Main Process
# Main Process Diagnostic Enhanced Version
def main():
    print("Masked FID calculation script started")
    print(f"Using device {DEVICE}")

    # 1 Load InceptionV3
    print("\nStep 1 of 6 Loading InceptionV3 model")
    try:
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.to(DEVICE)
        print("InceptionV3 model loaded successfully")
    except Exception as e:
        print(f"Error InceptionV3 model failed to load {e}")
        return

    # 2 Load images and masks
    print("\nStep 2 of 6 Loading images and masks")
    
    # Check if paths exist
    for dir_path in [REAL_DIR, FAKE_DIR, FAKE_DIR_base, MASK_DIR]:
        if not os.path.isdir(dir_path):
            print(f"Critical Error Directory does not exist {os.path.abspath(dir_path)}")
            print("Script terminated")
            return

    real_imgs = load_images(REAL_DIR, max_images=1020, transform=transform)
    # print(REAL_DIR)
    fake_imgs = load_images(FAKE_DIR, max_images=1020, transform=transform)
    # print(FAKE_DIR)
    fake_imgs_base = load_images(FAKE_DIR_base, max_images=1020, transform=transform)
    # print(FAKE_DIR_base)

    masks = load_mask(MASK_DIR,max_images=1020)
    # print(MASK_DIR)

    # Detailed check of loading results
    if real_imgs is None:
        print(f"Error Failed to load any real images from {REAL_DIR} Please check the path and file format")
        print("Script terminated")
        return
        
    if fake_imgs is None:
        print(f"Error Failed to load any generated images from {FAKE_DIR} Please check the path and file format")
        print("Script terminated")
        return

    if fake_imgs_base is None:
        print(f"Error Failed to load any generated images from {FAKE_DIR_base} Please check the path and file format")
        print("Script terminated")
        return

    if masks is None:
        print(f"Error Failed to load any mask images from {MASK_DIR} Please check the path and file format")
        print("Script terminated")
        return
    
    print(f"\nData loaded successfully {len(real_imgs)} real images {len(fake_imgs)} generated images {len(masks)} masks")

    # 3 Apply mask
    print("\nStep 3 of 6 Applying masks")
    try:
        # Only apply mask to generated images
        masked_fake_imgs = apply_mask(fake_imgs, masks)
        masked_fake_imgs_base = apply_mask(fake_imgs_base, masks)
        print("Masks applied successfully")
    except Exception as e:
        print(f"Error occurred while applying masks {e}")
        print(f"Please check if the number of generated images and masks match")
        print("Script terminated")
        return


    # 4 Extract features
    print("\nStep 4 of 6 Extracting features")
    real_feats = get_inception_features(real_imgs, inception)
    fake_feats = get_inception_features(masked_fake_imgs, inception)
    fake_feats_base = get_inception_features(masked_fake_imgs_base, inception)
    # fake_feats = get_inception_features(fake_imgs, inception)
    # fake_feats_base = get_inception_features(fake_imgs_base, inception)
    print("Feature extraction completed")

    # 5 Calculate mean and covariance
    print("\nStep 5 of 6 Calculating mean and covariance")
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    mu3, sigma3 = np.mean(fake_feats_base, axis=0), np.cov(fake_feats_base, rowvar=False)
    print("Calculation completed")

    # 6 Calculate Masked FID
    print("\nStep 6 of 6 Calculating Masked FID scores")
    mask_fid = calculate_mask_fid(mu1, sigma1, mu2, sigma2)
    mask_fid_base = calculate_mask_fid(mu1, sigma1, mu3, sigma3)
    print("\n")
    print(f"Mask FID Score from FAKE_DIR {round(mask_fid, 4)}")
    print(f"Mask FID Score from FAKE_DIR_base {round(mask_fid_base, 4)}")
    print("")
    print("Script execution finished")

if __name__ == "__main__":
    main()