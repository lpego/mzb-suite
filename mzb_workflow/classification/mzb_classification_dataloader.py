import torch
import numpy as np

from torchvision import utils
from PIL import Image, ImageFilter, ImageDraw, ImageOps

from torch.utils.data import Dataset


class MZBLoader(Dataset):
    def __init__(
        self,
        dir_dict,
        ls_inds=[],
        learning_set="all",
        transforms=None,
        glob_pattern="*_rgb.*",
    ):

        self.transforms = transforms
        self.imsize = 224

        self.ls_inds = ls_inds
        self.learning_set = learning_set
        self.glob_pattern = glob_pattern
        self.img_paths, self.labels, self.inds = self.prepare_data(
            dir_dict, ls_inds=self.ls_inds
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            with open(self.img_paths[idx], "rb") as f:
                img = Image.open(f).convert("RGB")
            label = self.labels[idx]

            if isinstance(self.transforms, dict):
                tensor_image = self.transforms[str(label.item())](img)
            else:
                tensor_image = self.transforms(img)

            return tensor_image, label, idx

        except OSError:
            return (
                torch.zeros((3, self.imsize, self.imsize)),
                torch.LongTensor([-1]).squeeze(),
                idx,
            )

    @staticmethod
    def prepare_data(
        dir_dict: dict, ls_inds: list = [], glob_pattern: str = "*_rgb.*"
    ) -> tuple:
        """Prepare data for training and testing, returns image paths, labels and indices
        Parameters:
            dir_dict (dict): dictionary with keys as class names and values as paths to images
            ls_inds (list): list of indices to be used for training or testing

        Returns:
            img_paths (list): list of image paths

        """
        # this makes a one folder - one class connection, and prepares data arrays consequently
        img_paths = []
        labels = []
        for i, key in enumerate(dir_dict):
            if isinstance(dir_dict[key], list):
                img = []
                for sub_dic in dir_dict[key]:
                    img += list(sub_dic.glob(glob_pattern))
            else:
                img = list(dir_dict[key].glob(glob_pattern))

            img_paths.extend(img)
            labels.extend(len(img) * [i])

        img_paths = np.asarray(img_paths, dtype=object)
        labels = np.asarray(labels)
        labels = torch.LongTensor(labels)

        if len(ls_inds) < 1:
            return img_paths, labels, ls_inds

        return img_paths[ls_inds], labels[ls_inds], ls_inds


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        x_n = tensor.mul_(self.std).add_(self.mean)
        return x_n

        # for t, m, s in zip(tensor, self.mean, self.std):
        #     x_n = t.mul_(s).add_(m)
        #     # The normalize code -> t.sub_(m).div_(s)
        # return x_n
