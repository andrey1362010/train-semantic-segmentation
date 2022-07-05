import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class TRAINS8K(Dataset):
    CLASSES = [
        'empty', 'secondary', 'main', 'trains'
    ]

    PALETTE = torch.tensor([
        [170, 66, 34], [250, 120, 120], [180, 250, 120], [6, 230, 250]
    ])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'training' if split == 'train' else 'validation'
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = -1

        img_path = Path(root) / 'images'
        self.files = list(img_path.glob('*.png'))
        if split == 'training': self.files = self.files[:8000]
        else: self.files = self.files[8000:]


        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'mask')

        image = io.read_image(img_path)
        label_mask = io.read_image(lbl_path)
        label = torch.zeros([1] + list(label_mask.shape[1:]))

        label[label_mask[:1, :, :] == 6] = 1
        label[label_mask[:1, :, :] == 7] = 2
        label[label_mask[:1, :, :] == 10] = 3

        #print(image.shape, label.shape)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label.squeeze().long()


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(TRAINS8K, '/media/andrey/big/datasets/hacksAI/TaskRZD/train')