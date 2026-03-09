import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ================== Configuration ==================
REAL_DIR = "./evaluate/real_images"      # real image path
FAKE_DIR = "./comparison_results_v4_10_1_rank16_all/lora"  # generated image path
BATCH_SIZE = 32
IMAGE_SIZE = 299
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== Data preprocessing ==================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def load_images(folder, max_images=None):
    """Load images as Tensor"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG")
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    paths = sorted(paths)
    if max_images:
        paths = paths[:max_images]

    imgs = []
    for p in tqdm(paths, desc=f"Loading {folder}"):
        with Image.open(p) as img:
            imgs.append(transform(img.convert("RGB")))
    return torch.stack(imgs), paths


# ================== KID calculation function ==================
def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
    """Polynomial Kernel, default parameters consistent with the paper"""
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + coef0) ** degree


def compute_mmd(x, y, degree=3, gamma=None, coef0=1):
    """Compute MMD^2 using polynomial kernel"""
    xx = polynomial_kernel(x, x, degree, gamma, coef0)
    yy = polynomial_kernel(y, y, degree, gamma, coef0)
    xy = polynomial_kernel(x, y, degree, gamma, coef0)

    m = x.size(0)
    n = y.size(0)

    mmd = (xx.sum() - torch.trace(xx)) / (m * (m - 1)) \
        + (yy.sum() - torch.trace(yy)) / (n * (n - 1)) \
        - 2 * xy.mean()
    return mmd.item()


def calculate_kid(real_feats, fake_feats, subset_size=100, n_subsets=10):
    """Compute KID as the average MMD over multiple random subsets"""
    kid_scores = []
    for _ in range(n_subsets):
        real_subset = real_feats[torch.randperm(real_feats.size(0))[:subset_size]]
        fake_subset = fake_feats[torch.randperm(fake_feats.size(0))[:subset_size]]
        kid = compute_mmd(real_subset, fake_subset)
        kid_scores.append(kid)
    return np.mean(kid_scores), np.std(kid_scores)


# ================== Feature extraction ==================
def get_inception_features(images, model):
    """Extract features from InceptionV3 (pool3 layer, 2048-dim)"""
    model.eval()
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Extracting features"):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            pred = model(batch)
            feats.append(pred.detach().cpu())
    return torch.cat(feats, dim=0)


# ================== Main program ==================
def main():
    print("Start calculating KID")
    print(f"Device: {DEVICE}")

    # 1. Load InceptionV3
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()  # remove classification head
    inception.to(DEVICE)

    # 2. Load data
    real_imgs, _ = load_images(REAL_DIR)
    fake_imgs, _ = load_images(FAKE_DIR)

    print(f"Loaded {len(real_imgs)} real images, {len(fake_imgs)} fake images")

    # 3. Extract features
    real_feats = get_inception_features(real_imgs, inception)
    fake_feats = get_inception_features(fake_imgs, inception)

    # 4. Compute KID
    kid_mean, kid_std = calculate_kid(real_feats, fake_feats)
    print(f"KID Score: {kid_mean:.6f} ± {kid_std:.6f}")


if __name__ == "__main__":
    main()