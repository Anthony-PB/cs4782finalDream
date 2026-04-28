import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DreamBoothDataset(Dataset):
    def __init__(self, instance_dir, class_dir, instance_prompt, class_prompt, size=512):
        self.instance_dir = instance_dir
        self.class_dir = class_dir
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.instance_images = [os.path.join(instance_dir, f) for f in os.listdir(instance_dir)]
        self.class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]

        self.num_instance_images = len(self.instance_images)
        self.num_class_images = len(self.class_images)

        # The length of the dataset is determined by the larger pool (the class images)
        self._length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Stream A: Instance Data (cycles through the 3-5 images using modulo)
        instance_image_path = self.instance_images[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path).convert("RGB")

        # Stream B: Prior/Class Data
        class_image_path = self.class_images[index % self.num_class_images]
        class_image = Image.open(class_image_path).convert("RGB")

        return {
            "instance_image": self.image_transforms(instance_image),
            "instance_prompt": self.instance_prompt,
            "class_image": self.image_transforms(class_image),
            "class_prompt": self.class_prompt
        }
