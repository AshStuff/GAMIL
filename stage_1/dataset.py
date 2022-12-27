import copy
import cv2
from torch.utils.data import Dataset
import albumentations as A

class AEDataset(Dataset):
    def __init__(self, data, transforms,  is_train=True):
        self.data = data
        self.transforms = transforms
        self.is_train = is_train

    def __getitem__(self, item):
        data_i = self.data[item]
        img = cv2.imread(data_i)
        data = {}
        data["image"] = img
        transform_out_1 = A.Compose(self.transforms)(**data)
        transform_out_2 = A.Compose(self.transforms)(**copy.deepcopy(data))
        return transform_out_1["image"].float(), transform_out_2["image"].float()

    def __len__(self):
        return len(self.data)