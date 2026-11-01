import torch
import torchvision

import utils.color_utils as utils

import numpy as np
from PIL import Image


# Define a custom transform to scale all channels by a random factor
class RandomScaling:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, img: Image.Image):
        rgb = np.array(img, dtype=np.float32)
        random_scale = np.random.rand(1)
        if self.dataset == "r":
            # set g, b channels to 0 in final image. Preserve r channel
            rgb[:, :, 1:] = 0
        elif self.dataset == "rg":
            rgb[:, :, 2] = 0
            rgb[:, :, np.random.randint(0, 2, (1,))] *= random_scale
        elif self.dataset == "blues":
            deleted_channel = np.random.randint(0, 2, (1,))
            rgb[:, :, deleted_channel] = 0
            modified_channel = np.random.choice([1 - deleted_channel[0], 2])
            rgb[:, :, modified_channel] *= random_scale
        elif self.dataset == "rgb":
            permutation = np.random.permutation(3)
            rgb[:, :, permutation[0]] *= random_scale
            rgb[:, :, permutation[1]] = 0
        else:
            raise ValueError("Invalid dataset. Select from r, rg, blues, rgb")
        return Image.fromarray(rgb.astype(np.uint8))


class RandomColor:
    def __init__(self):
        pass

    def __call__(self, img: Image.Image):
        # Randomly rotate hue
        hue_rotation = np.random.randint(0, 256)
        return utils.rotate_hue(img, hue_rotation, rgb_out=True)


class HueSeparation:
    def __init__(self, n_groups, rgb=True):
        self.n_groups = n_groups
        self.rgb = rgb

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        x_transformed = [
            torch.tensor(
                np.array(
                    utils.rotate_hue(img_hsv, (i * 256) // self.n_groups, rgb_out=self.rgb, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups)
        ]

        # x_transformed = [
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]

        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked


class SatSeparation:
    def __init__(self, n_groups, rgb=True):
        self.n_groups = n_groups
        self.rgb = rgb
        

    def __call__(self, img):
        img_hsv = img.convert("HSV")

        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_saturation(img_hsv,(2* i * 255) // self.n_groups, rgb_out=self.rgb, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups)
        ]

        # x_transformed = [
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]
        #x_transformed = x_transformed[2:] + x_transformed[:2]
        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked
    
class LumSeparation:
    def __init__(self, n_groups,amt, rgb=True):
        self.n_groups = n_groups
        self.rgb = rgb
        self.amt = amt

    def __call__(self, img):
        img_hsv = img.convert("HSV")

        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_luminance(img_hsv,self.amt+(2* i * 255) // self.n_groups, rgb_out=self.rgb, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups)
        ]

        # x_transformed = [
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]
        #x_transformed = x_transformed[2:] + x_transformed[:2]
        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked
    
class HueLuminanceSeparation:
    def __init__(self, n_groups, n_groups_saturation, rgb=True, frac_space=0.2):
        self.n_groups = n_groups
        self.n_groups_saturation = n_groups_saturation
        self.rgb = rgb
        self.frac_space = frac_space

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_saturation(
                    utils.rotate_hue(img_hsv, ((i // self.n_groups_saturation) * 256) // self.n_groups, rgb_out=False, rgb_in=False),
                    int((((i % self.n_groups_saturation) - self.n_groups_saturation // 2) / (self.n_groups_saturation // 2)) * 256 / 2 * self.frac_space), 
                        self.rgb, 
                        rgb_in=False
                        )
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups * self.n_groups_saturation)
        ]

        # x_transformed = [xw
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]

        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked


class TensorReshape:
    def __init__(self):
        pass

    def __call__(self, img):
        # Recall this will be applied to one image, so the output should be three (rather than four) dimensional
        return img.view(-1, img.shape[-2], img.shape[-1])

    
    
class HSVSeparation:
    def __init__(self, n_groups_hue, n_groups_saturation, n_groups_luminance,amt=0,amt_h=0,amt_s=0,amt_v=0, rgb=True, frac_space=1):
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.n_groups_luminance = n_groups_luminance
        self.rgb = rgb
        self.frac_space = frac_space
        self.amt=amt
        self.amt_h=amt_h
        self.amt_s=amt_s
        self.amt_v=amt_v

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_luminance(
                    utils.scale_saturation(
                    utils.rotate_hue(img_hsv, ((i % self.n_groups_hue) * 255) // self.n_groups_hue, rgb_out=False, rgb_in=False),
                    ((i % (self.n_groups_hue * self.n_groups_saturation))//self.n_groups_hue) *2* 255 // self.n_groups_saturation, rgb_out=False, rgb_in=False),
                    self.amt+(i//(self.n_groups_hue * self.n_groups_saturation)) *2* 255 // self.n_groups_luminance, rgb_out=False, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance)
        ]

        # x_transformed = [xw
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]

        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked

class HSVSeparation_V25:
    def __init__(self, n_groups_hue, n_groups_saturation, n_groups_luminance,amt=0, rgb=True, frac_space=1):
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.n_groups_luminance = n_groups_luminance
        self.rgb = rgb
        self.frac_space = frac_space
        self.amt=amt

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_luminance(
                    utils.scale_saturation2(
                    utils.rotate_hue(img_hsv, ((i % self.n_groups_hue) * 255) // self.n_groups_hue, rgb_out=False, rgb_in=False),
                    ((i % (self.n_groups_hue * self.n_groups_saturation))//self.n_groups_hue) *2* 255 // self.n_groups_saturation, rgb_out=False, rgb_in=False),
                    self.amt+(i//(self.n_groups_hue * self.n_groups_saturation)) *2* 255 // self.n_groups_luminance, rgb_out=False, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance)
        ]

        # x_transformed = [xw
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]

        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked
import random

class HSVSeparation_rand:
    def __init__(self, n_groups_hue, n_groups_saturation, n_groups_luminance,amt, rgb=True, frac_space=1):
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.n_groups_luminance = n_groups_luminance
        self.rgb = rgb
        self.frac_space = frac_space
        self.amt=amt

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        am=random.randint(0, 512)

        x_transformed = [
            torch.tensor(
                np.array(
                    utils.scale_luminance(
                    utils.scale_saturation(
                    utils.rotate_hue(img_hsv, ((i % self.n_groups_hue) * 255) // self.n_groups_hue, rgb_out=False, rgb_in=False),
                    ((i % (self.n_groups_hue * self.n_groups_saturation))//self.n_groups_hue) *2* 255 // self.n_groups_saturation, rgb_out=False, rgb_in=False),
                    am+(i//(self.n_groups_hue * self.n_groups_saturation)) *2* 255 // self.n_groups_luminance, rgb_out=False, rgb_in=False)
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)
            / 255
            for i in range(self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance)
        ]

        # x_transformed = [xw
        #     utils.rotate_hue_matrix(img, np.pi * 2 * i / self.n_groups)
        #     for i in range(self.n_groups)
        # ]

        x_stacked = torch.stack(
            x_transformed, dim=0
        )  # now (n_groups, 3, im_size, im_size)
        return x_stacked

