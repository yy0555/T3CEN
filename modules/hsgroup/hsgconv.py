"""
© Felix O'Mahony
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GroupConvHS(nn.Module):
    """
    Group Convolution Layer with hue and luminance group equivariance
    -----------------------
    This is a layer to be used within the global hue and luminance equivariance network. It is relatively simple, since no geometric transformation of the input tensor must take place. Rather, the input tensor has its group channels permuted so that each possible permutation of the color space (in hue, which we think of as rotation) and luminance (which we think of as scaling/permuting around radii of groups) occurs.

    This is described fully in the published paper.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        n_groups_hue=1,
        n_groups_saturation=1,
        n_groups_luminance=1,
        bias = False,
        rescale_luminance = True,
        ) -> None:
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if type(self.kernel_size) != int:
            self.kernel_size = self.kernel_size[0]  # we only permit square kernels

        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.n_groups_luminance = n_groups_luminance
        self.bias = bias

        self.rescale_luminance = rescale_luminance

        self.conv_weight = nn.Parameter(torch.Tensor(
            self.out_channels,
            self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance * self.in_channels,
            self.kernel_size,
            self.kernel_size
        ))

        # Initialize the weights
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))


    
    def forward_2(self, x):
        group_sizes = [self.n_groups_hue, self.n_groups_saturation, self.n_groups_luminance]

        # 取出大于1的维度（保持顺序），这些会被提前
        front_dims = [s for s in group_sizes if s > 1]

        # 补足到3个维度（被跳过的维度用1替代）
        while len(front_dims) < 3:
            front_dims.append(1)
        a, b, c = front_dims
        conv_weight = torch.zeros((a, b,c, self.out_channels, a, b,c,  self.in_channels, self.kernel_size, self.kernel_size), dtype=self.conv_weight.dtype)
        # put on same device as x
        conv_weight = conv_weight.to(x.device)
        cw = self.conv_weight.view(self.out_channels, a, b,c, self.in_channels, self.kernel_size, self.kernel_size)
        for i in range(self.n_groups_hue):
            for j in range(self.n_groups_saturation):
                for k in range(self.n_groups_luminance):
                    if self.n_groups_hue>1:
                        if self.n_groups_saturation>1:
                            if self.n_groups_luminance>1:
                                conv_weight[i, j, k, :, :, :, :, :, :, :] = cw.roll(i, dims=1).roll(j, dims=2).roll(k, dims=3)
                            else:
                                conv_weight[i, j, 0, :, :, :, :, :, :, :] = cw.roll(i, dims=1).roll(j, dims=2)
                        else:
                            if self.n_groups_luminance>1:
                                conv_weight[i, k, 0, :, :, :, :, :, :, :] = cw.roll(i, dims=1).roll(k, dims=2)
                            else:
                                conv_weight[i, 0, 0, :, :, :, :, :, :, :] = cw.roll(i, dims=1)
                    else:
                        if self.n_groups_saturation>1:
                            if self.n_groups_luminance>1:
                                conv_weight[j, k, 0, :, :, :, :, :, :, :] = cw.roll(j, dims=1).roll(k, dims=2)
                            else:
                                conv_weight[j, 0, 0, :, :, :, :, :, :, :] = cw.roll(j, dims=1)
                        else:
                            if self.n_groups_luminance>1:
                                conv_weight[k, 0, 0, :, :, :, :, :, :, :] = cw.roll(k, dims=1)
                            else:
                                conv_weight[0, 0, 0, :, :, :, :, :, :, :] = cw



                # the mask has four dimensions and is boolean
                    # we want to assign the values of conv_weight[a, b, :, c, d, :, :, :] to cw[:, i, j, :,:,:]
                    # where mask[a, b, c, d] is True
                # we can do this by reshaping conv_weight to have shape (n_groups * n_groups_luminance, out_channels, n_groups * n_groups_luminance, in_channels, kernel_size, kernel_size)
                # and then reshaping cw to have shape (out_channels, n_groups * n_groups_luminance, in_channels, kernel_size, kernel_size)
                    

        conv_weight = conv_weight.view(self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance* self.out_channels, self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance* self.in_channels, self.kernel_size, self.kernel_size)
        out_tensors = F.conv2d(x, conv_weight, stride=self.stride, padding=self.padding)
        return out_tensors


    
    def forward(self, x):
        # out_tensors = self.forward_1(x)
        out_tensors = self.forward_2(x)
        # out_tensors = self.forward_3(x)

        return out_tensors

class GroupPool(nn.Module):
    def __init__(
        self, n_groups_total, pool_operation=lambda x: torch.max(x, dim=1)[0], verbose=False, name=None
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.n_groups = n_groups_total
        if verbose:
            print("verbose is sent to True in Pooling Layer")
        self.name = name if name is not None else "GroupPool"
        self.pool_operation = pool_operation

    def forward(self, x):
        x = x.view(
            -1, self.n_groups, x.shape[1] // self.n_groups, x.shape[2], x.shape[3]
        )

        # incoming tensor is of shape (batch_size, n_groups * channels, height, width)
        # outgoing tensor should be of shape (batch_size, channels, height, width)
        y = self.pool_operation(x)

        return y

class GroupBatchNorm2d(nn.Module):
    def __init__(self, num_features, n_groups_hue=4, n_groups_saturation=1,n_groups_luminance=1, momentum = 0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, momentum=momentum)
        self.num_features = num_features
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.n_groups_luminance = n_groups_luminance

    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * channels, height, width)"""
        if x.shape[1] != self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance * self.num_features:
            raise ValueError(
                f"Expected {self.n_groups_hue * self.n_groups_saturation * self.num_features} channels in tensor, but got {x.shape[1]} channels"
            )
        x = x.view(
            -1, self.n_groups_hue, x.shape[-3] // (self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance), x.shape[-2], x.shape[-1]
        )
        x = x.permute(0, 2, 1, 3, 4)
        y = self.batch_norm(x)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(-1, (self.n_groups_hue * self.n_groups_saturation * self.n_groups_luminance) * self.num_features, x.shape[-2], x.shape[-1])
        return y

if __name__=="__main__":
    # NB input tensor has shape (batch, groups_hue * groups_saturation * channels, w, h)
    test_input = torch.randn(1, 4 * 3 * 3, 32, 32)
    group_conv = GroupConvHS(3, 3, 3, n_groups_hue=4, n_groups_saturation=3)
    group_conv(test_input)