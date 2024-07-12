import math

import torch
import torch.nn as nn
from torch import abs, sum


class MLP(nn.Module):
    """
    Simple module for projection MLPs
    """

    def __init__(self, input_dim=256, hidden_dim=2048, output_dim=256, no_biases=False):
        """
        :param input_dim: number of input units
        :param hidden_dim: number of hidden units
        :param output_dim: number of output units
        """
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=not no_biases),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=not no_biases),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    """
    Simple convolutional block with 3x3 conv filters used for VGG-like architectures
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pooling=True,
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
        no_biases=False,
    ):
        """
        :param in_channels (int):
        :param out_channels (int):
        :param pooling (bool):
        """

        super(ConvBlock, self).__init__()

        conv_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=not no_biases,
            groups=groups,
        )

        if pooling:
            pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            pool_layer = nn.Identity()

        self.module = nn.Sequential(conv_layer, nn.ReLU(inplace=True), pool_layer)

    def forward(self, x):
        return self.module(x)


class VGG11Encoder(nn.Module):
    """
    Custom implementation of VGG11 encoder with added support for greedy training
    """

    def __init__(
        self,
        train_end_to_end=False,
        projector_mlp=False,
        projection_size=256,
        hidden_layer_size=2048,
        base_image_size=32,
        no_biases=False,
        distance_top_down=1,
        error_correction=False,
        alpha_error=0.1,
        error_nb_updates=1,
    ):
        """
        :param train_end_to_end (bool): Enable backprop between conv blocks
        :param projector_mlp (bool): Whether to project representations through an MLP before calculating loss
        :param projection_size (int): Only used when projection mlp is enabled
        :param hidden_layer_size (int): Only used when projection mlp is enabled
        :param base_image_size (int): input image size (eg. 32 for cifar, 96 for stl10)
        """
        super(VGG11Encoder, self).__init__()

        # VGG11 conv layers configuration
        self.channel_sizes = [3, 64, 128, 256, 256, 512, 512, 512, 512]
        pooling = [True, True, False, True, False, True, False, True]

        # Configure end-to-end/layer-local architecture with or without projection MLP(s)
        self.layer_local = not train_end_to_end
        self.num_trainable_hooks = 1 if train_end_to_end else 8
        self.projection_sizes = (
            [projection_size] * self.num_trainable_hooks
            if projector_mlp
            else self.channel_sizes[-self.num_trainable_hooks :]
        )

        # Conv Blocks
        self.blocks = nn.ModuleList([])

        # Projector(s) - identity modules by default
        self.projectors = nn.ModuleList([])
        self.flattened_feature_dims = []

        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fm_sizes = [0]

        feature_map_size = base_image_size
        for i in range(8):
            if pooling[i]:
                feature_map_size /= 2
            self.fm_sizes.append(feature_map_size)
            self.blocks.append(
                ConvBlock(
                    self.channel_sizes[i],
                    self.channel_sizes[i + 1],
                    pooling=pooling[i],
                    no_biases=no_biases,
                )
            )
            input_dim = self.channel_sizes[i + 1]

            # Attach a projector MLP if specified either at every layer for layer-local training or just at the end
            if projector_mlp and (self.layer_local or i == 7):
                projector = MLP(
                    input_dim=int(input_dim),
                    hidden_dim=hidden_layer_size,
                    output_dim=projection_size,
                    no_biases=no_biases,
                )
                self.flattened_feature_dims.append(projection_size)
            else:
                projector = nn.Identity()
                self.flattened_feature_dims.append(
                    input_dim * feature_map_size * feature_map_size
                )
            self.projectors.append(projector)

        # Top-Down
        self.distance_top_down = distance_top_down
        self.links = [
            [i, i + self.distance_top_down]
            for i in range(1, 9 - self.distance_top_down)
        ]
        self.topdown_projectors = [
            self.initialize_topdown_projector(link) for link in self.links
        ]
        self.error_correction = error_correction
        # self.alpha_error = alpha_error
        self.alpha_errors = nn.ParameterList(
            [
                nn.Parameter(torch.ones(1) * alpha_error, requires_grad=True)
                for i in range(8)
            ]
        )
        self.T = error_nb_updates

    def initialize_topdown_projector(
        self, link: dict = None, hidden_dim: int = 512, no_biases: bool = False
    ):
        link_to, link_from = link

        in_channels = self.channel_sizes[link_from]
        out_channels = self.channel_sizes[link_to]
        in_dim = int(self.fm_sizes[link_from])
        out_dim = int(self.fm_sizes[link_to])

        # print(f"{in_dim=},{out_dim=},{in_channels=},{out_channels=}")
        stride = math.ceil(out_dim / in_dim)
        kernel_size = out_dim - stride * (in_dim - 1)
        padding = 0

        projector = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        return projector.cuda()

    def forward(self, x):
        z = []
        feature_maps = []

        if not self.error_correction:
            for i, block in enumerate(self.blocks):
                x = block(x)

                # For layer-local training, record intermediate feature maps and pooled layer activities z (after projection if specified)
                # Also make sure to detach layer outputs so that gradients are not backproped
                x_pooled = self.pooler(x).view(x.size(0), -1)
                z.append(self.projectors[i](x_pooled))
                feature_maps.append(x)

                if self.layer_local:
                    x = x.detach()
            x_pooled = self.pooler(x).view(x.size(0), -1)
            return x_pooled, feature_maps, z

        else:
            pred_error = [0]
            # first pass
            x_current = self.blocks[0](x)
            x_pooled = self.pooler(x_current).view(x_current.size(0), -1)
            z.append(self.projectors[0](x_pooled))
            feature_maps.append(x_current)
            if self.layer_local:
                x_current = x_current.detach()

            # intermediate layers get corrected
            for i in range(1, 7):
                x_next = self.blocks[i](x_current)
                error = torch.tensor(0)
                for t in range(self.T):
                    prediction = self.topdown_projectors[i - 1](x_next)
                    error = x_current - prediction
                    # bn = nn.BatchNorm2d(error.shape[1]).cuda()
                    # error = bn(error)
                    # x_current = x_current - abs(self.alpha_error * error)
                    # x_next = self.blocks[i](x_current)
                    x_next = x_next + self.alpha_errors[i] * self.blocks[i](error)

                pred_error.append(sum(abs(error)))
                _next_pooled = self.pooler(x_next).view(x_next.size(0), -1)
                z.append(self.projectors[i](_next_pooled))
                feature_maps.append(x_next)
                if self.layer_local:
                    x_next = x_next.detach()
                x_current = x_next

            # last layer cannot receive top-down feedback
            x_last = self.blocks[-1](x_next)
            x_pooled = self.pooler(x_last).view(x_last.size(0), -1)

            return x_pooled, feature_maps, z, pred_error
