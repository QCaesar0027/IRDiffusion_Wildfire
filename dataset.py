import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def compute_gradient_info(ir_tensor):
    """
    Compute gradient information from the infrared image to capture fire boundaries and spread direction
    
    Args:
        ir_tensor: [C, H, W] infrared intensity image tensor
    
    Returns:
        gradient_magnitude: [C, H, W] gradient magnitude map representing the intensity of changes
    """
    # Sobel convolution kernel for X direction vertical edge detection
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=ir_tensor.dtype, device=ir_tensor.device).view(1, 1, 3, 3)
    
    # Sobel convolution kernel for Y direction horizontal edge detection
    sobel_y = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=ir_tensor.dtype, device=ir_tensor.device).view(1, 1, 3, 3)
    
    # Process input tensor dimensions
    if len(ir_tensor.shape) == 2:  # [H, W] -> [1, 1, H, W]
        ir_input = ir_tensor.unsqueeze(0).unsqueeze(0)
        need_squeeze = True
    elif len(ir_tensor.shape) == 3:  # [C, H, W] -> [1, C, H, W]
        ir_input = ir_tensor.unsqueeze(0)
        need_squeeze = True
    else:  # [B, C, H, W]
        ir_input = ir_tensor
        need_squeeze = False
    
    # If it is multi channel compute gradients for each channel separately
    if ir_input.shape[1] > 1:
        # Compute gradients for each channel separately
        grad_x_list = []
        grad_y_list = []
        for c in range(ir_input.shape[1]):
            single_channel = ir_input[:, c:c+1, :, :]  # [B, 1, H, W]
            grad_x_c = F.conv2d(single_channel, sobel_x, padding=1)
            grad_y_c = F.conv2d(single_channel, sobel_y, padding=1)
            grad_x_list.append(grad_x_c)
            grad_y_list.append(grad_y_c)
        grad_x = torch.cat(grad_x_list, dim=1)  # [B, C, H, W]
        grad_y = torch.cat(grad_y_list, dim=1)  # [B, C, H, W]
    else:
        # Single channel case
        grad_x = F.conv2d(ir_input, sobel_x, padding=1)
        grad_y = F.conv2d(ir_input, sobel_y, padding=1)
    
    # Compute gradient magnitude sqrt(grad_x^2 + grad_y^2)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # Add a small value to avoid sqrt(0)

    gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
    gradient_magnitude = torch.pow(gradient_magnitude, 0.7)
    gradient_magnitude = gradient_magnitude.clamp(0.0, 1.0)
    
    # Restore original dimensions
    if need_squeeze:
        gradient_magnitude = gradient_magnitude.squeeze(0)  # Remove batch dimension
    
    return gradient_magnitude

class WildfireDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        resolution=512,
        ir_threshold=0.2, # Compatibility parameter kept but not directly used for adaptive threshold
        # New dataset split support
        split=None,  # train val test or None use all data
        splits_file="dataset_splits.json",  # Split configuration file
        # New adjustable quantile thresholds and postprocessing parameters
        q_high=0.95,
        q_low=0.05,
        morph_kernel=3,
        local_kernel=5,
        local_thresh=0.2,
        # New configurable text prompt
        text_prompt="A cinematic, high-resolution, ultra-detailed photograph of a realistic wildfire flame",
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.ir_threshold = ir_threshold
        self.split = split
        self.splits_file = splits_file
        self.q_high = q_high
        self.q_low = q_low
        self.morph_kernel = morph_kernel
        self.local_kernel = local_kernel
        self.local_thresh = local_thresh
        self.text_prompt = text_prompt
        
        # Set RGB and IR folder paths corrected paths
        self.rgb_path = os.path.join(data_dir, "processed_rgb")
        self.ir_path = os.path.join(data_dir, "thermal")
        
        # Load file list supports dataset split
        self.image_files = self._load_image_files()
        
        # Define the standard image preprocessing pipeline
        self.rgb_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Normalize pixel values from [0, 1] to [-1, 1]
        ])
        
        # Apply similar processing to IR image which is usually single channel
        self.ir_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST), # Use NEAREST for mask like information
            transforms.ToTensor(),
        ])

        # Precompute text tokens to avoid repeated computation for each sample
        if self.tokenizer is not None:
            self.text_input_ids = self.tokenizer(
                self.text_prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]
        else:
            self.text_input_ids = None
    
    def _load_image_files(self):
        """
        Load the image file list with dataset split support
        
        Returns:
            list: a list containing tuples of rgb_file and ir_file
        """
        import json
        
        # If a split is specified and the split file exists use the split list
        if self.split and os.path.exists(self.splits_file):
            print(f"Loading dataset split: {self.split} from {self.splits_file}")
            
            with open(self.splits_file, 'r', encoding='utf-8') as f:
                splits_config = json.load(f)
            
            if self.split in splits_config['splits']:
                file_pairs = []
                for item in splits_config['splits'][self.split]:
                    rgb_file = item['rgb']
                    ir_file = item['ir']
                    
                    # Check whether files exist
                    rgb_full_path = os.path.join(self.rgb_path, rgb_file)
                    ir_full_path = os.path.join(self.ir_path, ir_file)
                    
                    if os.path.exists(rgb_full_path) and os.path.exists(ir_full_path):
                        file_pairs.append((rgb_file, ir_file))
                
                print(f"Loaded {self.split} set: {len(file_pairs)} file pairs")
                return file_pairs
            else:
                print(f"Split '{self.split}' not found in split file falling back to automatic matching mode")
        
        # Fall back to the original automatic matching mode
        print("Using automatic matching mode to load all matched file pairs")
        
        def _list_images(p):
            if not os.path.isdir(p):
                return set()
            return {f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
        
        rgb_files = _list_images(self.rgb_path)
        ir_files = _list_images(self.ir_path)
        
        # Method 1 direct filename matching 00001.JPG <-> 00001.JPG
        common_files = rgb_files & ir_files
        file_pairs = [(f, f) for f in common_files]
        
        # Method 2 number matching MAX_0528.JPG <-> IRX_0528.jpg
        rgb_by_number = {}
        ir_by_number = {}
        
        for f in rgb_files:
            if f.startswith('MAX_'):
                try:
                    number = f.split('_')[1].split('.')[0]
                    rgb_by_number[number] = f
                except IndexError:
                    continue
        
        for f in ir_files:
            if f.startswith('IRX_'):
                try:
                    number = f.split('_')[1].split('.')[0]
                    ir_by_number[number] = f
                except IndexError:
                    continue
        
        # Add number matched file pairs
        for number in rgb_by_number:
            if number in ir_by_number:
                file_pairs.append((rgb_by_number[number], ir_by_number[number]))
        
        print(f"Automatic matching found {len(file_pairs)} file pairs")
        return sorted(file_pairs)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load original images
        attempts = 0
        while attempts < 3:
            rgb_filename, ir_filename = self.image_files[idx]
            try:
                rgb_image = Image.open(os.path.join(self.rgb_path, rgb_filename)).convert("RGB")
                ir_image = Image.open(os.path.join(self.ir_path, ir_filename)).convert("L") # L means loading as single channel grayscale
                break
            except Exception as e:
                print(f"Error loading image pair ({rgb_filename}, {ir_filename}): {e}")
                idx = (idx + 1) % len(self.image_files)
                attempts += 1
        else:
            raise RuntimeError("Failed to load images after multiple attempts.")

        # Apply base transform and convert to Tensor
        rgb_tensor = self.rgb_transform(rgb_image) # shape: (3, H, W), range: [-1, 1]
        ir_tensor = self.ir_transform(ir_image)   # shape: (1, H, W), range: [0, 1]

        # Flame color scheme separation of responsibilities
        # Control signal controlnet_hint pure flame features without position
        #   R channel linear intensity similar to flame temperature
        #   G channel intensity squared high temperature region enhancement similar to flame brightness
        #   B channel reserved extension 0
        # 
        # Position mask mask provided separately to the inpainting pipeline
        #   Clearly indicates where to generate flames
        # 
        # This separation makes the responsibilities clearer and allows ControlNet to focus on learning flame features rather than position

        # Signal A soft mask
        # Use quantile adaptive thresholding to generate a soft mask for fire location
        ir_small = F.interpolate(ir_tensor.unsqueeze(0), scale_factor=0.25, mode='area').squeeze(0)
        flat = ir_small.flatten()
        
        # Safe quantile computation to avoid NaN
        if len(flat) > 0 and not torch.isnan(flat).all():
            # Remove NaN values
            valid_flat = flat[~torch.isnan(flat)]
            if len(valid_flat) > 0:
                thr = torch.quantile(valid_flat, self.q_high)
            else:
                thr = torch.tensor(0.5)  # Default threshold
        else:
            thr = torch.tensor(0.5)  # Default threshold
        
        # Create soft mask use a smooth transition near the threshold instead of hard binarization
        soft_mask = torch.sigmoid((ir_tensor - thr) * 10)  # 10 is the sharpening parameter
        
        # Ensure soft_mask has no NaN
        soft_mask = torch.nan_to_num(soft_mask, nan=0.0)
        
        # Morphological processing on the soft mask
        def _dilate(x, k: int = 3):
            return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

        def _erode(x, k: int = 3):
            return 1 - F.max_pool2d(1 - x, kernel_size=k, stride=1, padding=k // 2)

        soft_mask = _dilate(_erode(soft_mask, self.morph_kernel), self.morph_kernel)

        # Small region suppression requires the local 5x5 average to reach at least a certain ratio
        pad = self.local_kernel // 2
        local_avg = F.avg_pool2d(soft_mask, kernel_size=self.local_kernel, stride=1, padding=pad)
        soft_mask = (local_avg >= self.local_thresh).float() * soft_mask

        # Flame intensity signal
        # Normalize intensity to [0,1] as an encoding of flame temperature
        if len(valid_flat) > 0:
            p_low = torch.quantile(valid_flat, self.q_low)
            p_high = torch.quantile(valid_flat, 0.99)
        else:
            p_low = torch.tensor(0.0)
            p_high = torch.tensor(1.0)
        
        if p_high > p_low + 1e-6:
            # Linear normalization preserving the original intensity relationship
            intensity_linear = (ir_tensor - p_low) / (p_high - p_low)
            intensity_linear = intensity_linear.clamp(0.0, 1.0)
        else:
            intensity_linear = torch.zeros_like(ir_tensor)
        
        # Ensure there is no NaN
        intensity_linear = torch.nan_to_num(intensity_linear, nan=0.0)
        
        # Note do not multiply with soft_mask here so that intensity information remains complete
        # Position information is provided separately by mask
        
        # Flame color encoding
        
        # R channel linear intensity base flame temperature
        flame_r = intensity_linear
        
        # G channel squared intensity high temperature region enhancement similar to the bright flame core
        # This makes high temperature regions 0.8²=0.64 more prominent and low temperature regions 0.2²=0.04 darker
        flame_g = torch.pow(intensity_linear, 2.0)
        
        # B channel reserved extension currently set to 0 future information can be added here
        flame_b = torch.zeros_like(intensity_linear)
        
        # Channel packing
        # Create a 3 channel hint similar to a flame heatmap
        # With this encoding
        #   High temperature region (1.0, 1.0, 0) similar to yellow
        #   Medium temperature region (0.6, 0.36, 0) similar to orange red
        #   Low temperature region (0.2, 0.04, 0) similar to dark red
        controlnet_hint = torch.cat([flame_r, flame_g, flame_b], dim=0)
        
        # Final safety check
        controlnet_hint = torch.nan_to_num(controlnet_hint, nan=0.0)

        # Prepare CLIP text guidance

        # Use precomputed text tokens if tokenizer is None return None
        text_input_ids = self.text_input_ids if hasattr(self, 'text_input_ids') else None

        # Return a dictionary containing all information
        return {
            "rgb_image": rgb_tensor,
            "controlnet_hint": controlnet_hint,  # Flame color encoding [I, I², 0]
            "mask": soft_mask,                    # Position mask provided separately
            "text_input_ids": text_input_ids,
            "filename": rgb_filename,
            "rgb_filename": rgb_filename,
            "ir_filename": ir_filename
        }