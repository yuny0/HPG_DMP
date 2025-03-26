import numpy as np
from torch.utils import data as data
import os
import cv2
import numpy as np
import math
from basicsr.data.bsrgan_util import degradation_bsrgan
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from .data_util import make_dataset
from numpy.random import RandomState

import cv2
import random
import torch



def grad(src):
    sobelx = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    detail_grad = sobelxy
    return detail_grad



import numpy as np
import cv2
import math
from torchvision.transforms import Compose


def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):

        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample




def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)



def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


def random_crop(img, out_size):
    img_ori = img
    h, w = img.shape[:2]
    if h - out_size < 0 or w - out_size < 0:
        img_ = np.zeros([3 * h, 3 * w, 3])
        # print('img_',img_.shape)
        # print('img_ori', img_ori.shape)
        img_[0:h, 0:w, :] = img_ori[:, :, :]
        img_[h:2 * h, 0:w, :] = img_ori[:, :, :]
        img_[2 * h:3 * h, 0:w, :] = img_ori[:, :, :]
        img_ori = img_
        img_[:, 0:w, :] = img_ori[:, 0:w, :]
        img_[:, w:2 * w, :] = img_ori[:, 0:w, :]
        img_[:, 2 * w:3 * w, :] = img_ori[:, 0:w, :]
        h, w = img_.shape[:2]
        rnd_h = random.randint(0, h - out_size)
        rnd_w = random.randint(0, w - out_size)
        # print('img_[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]',img_[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size])
        return img_[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]
    else:
        h, w = img.shape[:2]
        rnd_h = random.randint(0, h - out_size)
        rnd_w = random.randint(0, w - out_size)
        return img[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]


@DATASET_REGISTRY.register()
class BSRGANTrainDataset(data.Dataset):
    def __init__(self, opt):
        super(BSRGANTrainDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.rand_state = RandomState(0)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.gt_paths = make_dataset(self.gt_folder)

        self.model_type = opt['model_type']

        self.midas_transforms = self.transform(model_type=self.model_type)

    def transform(self, model_type):
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            dpt_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    PrepareForNet(),
                    lambda sample: torch.from_numpy(sample["image"]),
                ]
            )
            return dpt_transform
        elif model_type == "MiDaS_small":
            small_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        256,
                        256,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    lambda sample: torch.from_numpy(sample["image"]),
                ]
            )
            return small_transform
        else:
            default_transform = Compose(
                [
                    lambda img: {"image": img / 255.0},
                    Resize(
                        384,
                        384,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="upper_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                    lambda sample: torch.from_numpy(sample["image"]),
                ]
            )

            return default_transform

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        lq_path = self.lq_folder + '/' + gt_path.split('/')[-1]
        img_gt = cv2.imread(gt_path)
        img_lq = cv2.imread(lq_path)

        img_pair = []
        img_pair.append(img_lq)
        img_pair.append(img_gt)
        O, B = self.crop(img_pair)


        img_gt = B.astype(np.float32)
        img_lq = O.astype(np.float32)

        img_gt = img_gt[:, :, [2, 1, 0]]
        img_lq = img_lq[:, :, [2, 1, 0]]

        gt_size = self.opt['gt_size']

        if self.opt['use_flip']:
            img_gt, img_lq = self.flip([img_gt, img_lq])
        if self.opt['use_rot']:
            img_gt, img_lq = self.rotate([img_gt, img_lq])

        transform_ours = self.transform(self.model_type)
        depth_transform_gt = transform_ours(img_gt)
        depth_transform_lq = transform_ours(img_lq)


        img_gt = img_gt / 255.
        img_lq = img_lq / 255.

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'depth_transform_gt': depth_transform_gt,
            'depth_transform_lq': depth_transform_lq,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)

    def crop(self, img_pair, aug=False):
        patch_size = self.opt['gt_size']
        h, w, c = img_pair[0].shape

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi = 1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size
        if h - p_h <= 0 or w - p_w <= 0:
            O = cv2.resize(img_pair[0], (patch_size, patch_size))
            B = cv2.resize(img_pair[1], (patch_size, patch_size))
            return O, B
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
            O = img_pair[0][r: r + p_h, c: c + p_w, :]
            B = img_pair[1][r: r + p_h, c: c + p_w, :]
            return O, B

    def flip(self, imgs):
        out = imgs
        if self.rand_state.rand() > 0.5:
            out = []
            for img in imgs:
                tmp = np.flip(img, axis=1)
                out.append(tmp)

        return out

    def rotate(self, imgs):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.opt['gt_size']
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        out = []
        for img in imgs:
            tmp =cv2.warpAffine(img, M, (patch_size, patch_size))
            out.append(tmp)
        return out

