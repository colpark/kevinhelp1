# src/data/dataset/cifar10.py
"""
CIFAR-10 dataset wrappers for PixNerDiT training.
"""
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor


class PixCIFAR10(CIFAR10):
    """
    CIFAR-10 dataset wrapper for PixNerDiT training.

    Returns normalized images in [-1, 1] range with metadata.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        random_flip: bool = True,
        download: bool = False,
    ):
        super().__init__(root=root, train=train, download=download)

        transform_list = [transforms.ToTensor()]
        if random_flip and train:
            transform_list.append(transforms.RandomHorizontalFlip())

        self.base_transform = transforms.Compose(transform_list)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)

        # Apply base transforms
        if self.base_transform is not None:
            raw_image = self.base_transform(image)
        else:
            raw_image = to_tensor(image)

        # Normalize to [-1, 1]
        normalized_image = self.normalize(raw_image)

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return normalized_image, target, metadata


class CIFAR10RandomNDataset(torch.utils.data.Dataset):
    """
    Random noise dataset for evaluation and sampling.

    Generates random noise and class labels for unconditional/conditional sampling.
    Similar to ImageNet's randn.py but for CIFAR-10.
    """
    def __init__(
        self,
        num_classes: int = 10,
        latent_shape: tuple = (3, 32, 32),
        max_num_instances: int = 1000,
    ):
        self.num_classes = num_classes
        self.latent_shape = latent_shape
        self.max_num_instances = max_num_instances

    def __len__(self):
        return self.max_num_instances

    def __getitem__(self, idx: int):
        # Generate random noise
        latent = torch.randn(self.latent_shape)

        # Cycle through classes
        target = idx % self.num_classes

        metadata = {
            "class": target,
        }
        return latent, target, metadata
