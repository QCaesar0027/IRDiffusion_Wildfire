# Infrared-Guided Diffusion Models for Realistic Wildfire Image Synthesis

**Shichang Xu Cao, Pengle Cheng, and Juan Liu<sup>*</sup>**

---

##  Environment

This project was developed with the following environment:

- Python 3.9
- PyTorch 2.5.1
- CUDA 12.1
- diffusers
- transformers

---

## Create Conda Environment

Clone the repository:

```bash
git clone https://github.com/QCaesar0027/IRDiffusion_Wildfire.git
cd IRDiffusion_Wildfire
```

---

## Dataset

You can download the datasets from the following links:
After downloading and extracting the files, place them in the **project root directory**.

- [Wildfire Dataset](https://pan.baidu.com/s/1icvJXJB4gMyk7J0AaqCziQ?pwd=jc49)  
  Extraction code: **jc49**

  Description:

- **data/fire_pairs_new** – Paired wildfire images used for diffusion model training  
- **data/processed_rgb** – Processed RGB wildfire images  
- **data/thermal** – Infrared / thermal images used for guidance
---
- **mask_info_v2** - mask info


---

## Fire Detection Model

- [Detection Model](https://pan.baidu.com/s/1NubC0OG3_elkGZCA04gOTQ?pwd=7xe5)  
  Extraction code: **7xe5**

  Description:
The second download link contains a trained **YOLO fire detection model**:

---

##  Training

To train the infrared-guided diffusion model(LoRA fine-tuning), for example, run:

```bash
python train.py   --instance_data_dir ./data/fire_pairs_new   --output_dir ./lora_unet_output_anti_yellow_v2_16   --train_batch_size 4   --max_train_steps 9000   --learning_rate 1e-4   --lora_rank 16  --lora_alpha 32
```

---

##  Testing

To test the trained model and generate wildfire images, for example, run:

```bash
python -u test.py   --base-model runwayml/stable-diffusion-v1-5   --compare-pretrained   --batch-test-set      --output-dir comparison_results_v4_10_1_rank16_all  --guidance 10 --strength 1.0   --control-type rgb   --lora-path ./lora_unet_output_anti_yellow_v2_16/best_lora.safetensors
```


##  Comparison experiment

To run the comparison experiments, for example, use:

```bash
python test_baseline.py \
  --batch-test-set \
  --output-dir "./images_hed_sd" \
  --base-model "runwayml/stable-diffusion-v1-5" \
  --pretrained-controlnet-id "lllyasviel/sd-controlnet-hed"
```
---

##  Evaluate

clip, fid, LPIPS, psnr .....

---

##  YOLO Fire Detection Evaluation

This project uses a trained **YOLO fire detection model** to evaluate generated wildfire images.

###  Run Fire Detection

Run the detection script:

```bash
python fire_detection.py



