import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os


class PaintingDataset(Dataset):
    def __init__(self, json_path, time=11, is_image=False):
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.is_image = is_image
        self.time = time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_folder_path = self.data[idx]["images_path"]

        if not os.path.isdir(img_folder_path):
            raise ValueError(f"error: path {img_folder_path}")

        image_files = [os.path.join(img_folder_path, f) for f in os.listdir(img_folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
        if not image_files:
            raise ValueError(f"error: No image in {img_folder_path}")

        images = []
        for img_path in image_files:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        real_len = len(images)
        padding_mask = torch.zeros(self.time, dtype=torch.long)
        padding_mask[:real_len] = 1

        if len(images) < self.time:
            pad_size = self.time - len(images)
            pad_tensor = torch.zeros((3, 256, 256))
            images.extend([pad_tensor.clone() for _ in range(pad_size)])  
        else:
            images = images[:self.time]

        images = torch.stack(images) 
        
        if self.is_image:
            valuation_results = self.data[idx]["valuationResults"]
            labels = torch.tensor([
                valuation_results["drawingAccuracy"]["imageConsistency"] / 10,
                valuation_results["drawingStability"]["styleStability"] / 10,
                valuation_results["drawingStability"]["colorStability"] / 10,
                valuation_results["drawingStability"]["compositionStability"] / 10,
                valuation_results["drawingStability"]["processStability"] / 10,
                valuation_results["drawingDepth"]["detailEnhancement"] / 10,
                valuation_results["drawingDepth"]["colorDevelopment"] / 10,
                valuation_results["drawingDepth"]["compositionComplexity"] / 10
            ], dtype=torch.float32)
        else:
            valuation_results = self.data[idx]["valuationResults"]
            labels = torch.tensor([
                valuation_results["drawingAccuracy"]["promptConsistency"] / 10,
                valuation_results["drawingStability"]["styleStability"] / 10,
                valuation_results["drawingStability"]["colorStability"] / 10,
                valuation_results["drawingStability"]["compositionStability"] / 10,
                valuation_results["drawingStability"]["processStability"] / 10,
                valuation_results["drawingDepth"]["detailEnhancement"] / 10,
                valuation_results["drawingDepth"]["colorDevelopment"] / 10,
                valuation_results["drawingDepth"]["compositionComplexity"] / 10
            ], dtype=torch.float32)

        return images, padding_mask, labels
