import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import os

loss_fn = lpips.LPIPS(net='alex').cuda()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def load_img(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0) * 2 - 1
    return img.cuda()

def compute_lpips_dir(gen_dir, ref_dir):
    gen_files = sorted(glob(os.path.join(gen_dir, "*")))
    ref_files = sorted(glob(os.path.join(ref_dir, "*")))

    scores = []
    for g, r in zip(gen_files, ref_files):
        img_g = load_img(g)
        img_r = load_img(r)

        with torch.no_grad():
            lp = loss_fn(img_g, img_r)
            scores.append(float(lp))

    print("Average LPIPS:", sum(scores)/len(scores))
    return scores


compute_lpips_dir("./comparison_results_v4_10_1_rank16_all_I200/lora", "./data/processed_rgb")