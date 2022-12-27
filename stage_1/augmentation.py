import torch
import numpy as np
import albumentations as A
from albumentations import transforms
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2

class Normalize(ImageOnlyTransform):
    def __init__(self, min, max, always_apply=True, p=1):
        super(Normalize, self).__init__(always_apply=always_apply, p=p)
        self.min = min
        self.max = max
    def apply(self, x, **params):
        minVal = np.percentile(x, self.min)
        maxVal = np.percentile(x, self.max)
        x[x > maxVal] = maxVal
        x[x < minVal] = minVal
        return x

class NormalizeIntensity(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1):
        super(NormalizeIntensity, self).__init__(always_apply=always_apply, p=p)

    def apply(self, x, **params):
        std_val = np.std(x)
        mean_val = np.mean(x)
        x = (x - mean_val)/std_val
        return x

class Transforms():
    @property
    def train_transform(self):
        return [
            transforms.ChannelShuffle(p=0.5),
            transforms.CLAHE(p=0.5),
            transforms.ColorJitter(p=0.5),
            A.GaussNoise(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            Normalize(min=1, max=99),
            NormalizeIntensity(),
            ToTensorV2(),
        ]
    @property
    def val_transform(self):
        return [
            Normalize(min=1, max=99),
            NormalizeIntensity(),
            ToTensorV2(),

        ]