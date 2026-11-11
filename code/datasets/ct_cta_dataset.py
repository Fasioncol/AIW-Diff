import os
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from skimage.filters.rank import entropy
from skimage.morphology import disk

def dcm_to_tensor(dcm_path, target_size=(512, 512)):
    dcm_file = pydicom.dcmread(dcm_path)
    pixel_array = dcm_file.pixel_array
    if hasattr(dcm_file, 'RescaleSlope') and hasattr(dcm_file, 'RescaleIntercept'):
        slope = float(dcm_file.RescaleSlope)
        intercept = float(dcm_file.RescaleIntercept)
        hu_image = pixel_array * slope + intercept
    else:
        hu_image = pixel_array

    body_mask = hu_image > -500
    body_pixels = hu_image[body_mask]

    lower_bound = np.percentile(body_pixels, 1)
    upper_bound = np.percentile(body_pixels, 99)
    clipped_hu = np.clip(hu_image, lower_bound, upper_bound)

    normalized_img = ((clipped_hu - lower_bound) / (upper_bound - lower_bound)) * 255.0
    normalized_img = normalized_img.astype(np.uint8)

    entropy_map = entropy(normalized_img, disk(5))

    ent_min = entropy_map.min()
    ent_max = entropy_map.max()
    if ent_max > ent_min:
        entropy_map = (entropy_map - ent_min) / (ent_max - ent_min)
    else:
        entropy_map = np.zeros_like(entropy_map)

    entropy_map = entropy_map.astype(np.float32)

    tensor = torch.from_numpy(normalized_img).unsqueeze(0).float()
    complexity_tensor = torch.from_numpy(entropy_map).unsqueeze(0).float()

    tensor_resized = F.resize(tensor, target_size, antialias=True)
    complexity_tensor_resized = F.resize(complexity_tensor, target_size, antialias=True)

    return tensor_resized, complexity_tensor_resized

class CT_CTADataset(Dataset):
    def __init__(self, ct_dir, cta_dir, image_size):
        if image_size != 512:
            raise ValueError(f"本项目只支持 512×512 训练和验证，当前 image_size={image_size}")

        self.ct_dir = ct_dir
        self.cta_dir = cta_dir
        self.image_size = image_size

        self.ct_files = sorted([f for f in os.listdir(self.ct_dir) if f.endswith('.dcm')])
        self.cta_files = sorted([f for f in os.listdir(self.cta_dir) if f.endswith('.dcm')])

        assert len(self.ct_files) == len(
            self.cta_files), f"CT和CTA文件数量不匹配！\nCT文件夹: {len(self.ct_files)} 张\nCTA文件夹: {len(self.cta_files)} 张"
        assert len(self.ct_files) > 0, f"在 {self.ct_dir} 中未找到DCM文件！"

        print(f"成功找到 {len(self.ct_files)} 对CT/CTA图像。")

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        ct_file_name = self.ct_files[idx]
        cta_file_name = self.cta_files[idx]

        ct_path = os.path.join(self.ct_dir, ct_file_name)
        cta_path = os.path.join(self.cta_dir, cta_file_name)

        ct_tensor, ct_complexity_map = dcm_to_tensor(ct_path, target_size=(self.image_size, self.image_size))

        cta_tensor, _ = dcm_to_tensor(cta_path, target_size=(self.image_size, self.image_size))

        return ct_tensor, cta_tensor, ct_complexity_map
