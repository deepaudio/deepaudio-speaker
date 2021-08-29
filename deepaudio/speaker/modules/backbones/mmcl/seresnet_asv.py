import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer

from .resnet import ResNet, BasicBlock, ResLayer
from .STP import StatsPooling
from mmcls.models.utils import SELayer
import torch.utils.checkpoint as cp


class SEBasicBlock(BasicBlock):
    """SEBasicBlock block for SEResNet.

    Args:
        in_channels (int): The input channels of the SEBasicBlock block.
        out_channels (int): The output channel of the SEBasicBlock block.
        se_ratio (int): Squeeze ratio in SELayer. Default: 16

    """

    def __init__(self, in_channels, out_channels, se_ratio=16, **kwargs):
        super(SEBasicBlock, self).__init__(in_channels, out_channels, **kwargs)
        self.se_layer = SELayer(out_channels, ratio=se_ratio, act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')))

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            out = self.se_layer(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class SEResNet_SV(ResNet):
    """SEResNet backbone for speaker verification.

    Compared to standard ResNet, it uses `kernel_size=3` and `stride=1` in
    conv1, and does not apply MaxPoolinng after stem. It has been proven to
    be more efficient than standard ResNet in other public codebase, e.g.,
    `https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py`.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        se_ratio (int): compression ratio, from {2, 4, 8, 16}, default=4
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): This network has specific designed stem, thus it is
            asserted to be False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        18: (SEBasicBlock, (2, 2, 2, 2)),
        34: (SEBasicBlock, (3, 4, 6, 3))
    }

    def __init__(self, depth, input_dim=40, se_ratio=16, deep_stem=False, **kwargs):
        self.se_ratio = se_ratio
        super(SEResNet_SV, self).__init__(
            input_dim, depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if len(x.shape) == 3:
            # audio format
            x.unsqueeze_(1)
            x = x.transpose(2, 3)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def make_res_layer(self, **kwargs):
        return ResLayer(se_ratio=self.se_ratio, **kwargs)


def MainModel(configs):
    arch = SEResNet_SV(
        input_dim=configs.feature.n_mels,
        depth=configs.model.depth,
        in_channels=configs.model.in_channels,
        stem_channels=configs.model.stem_channels,
        base_channels=configs.model.base_channels,
        num_stages=configs.model.num_stages,
        out_indices=(configs.model.out_indices,),
        norm_cfg=dict(type=configs.model.norm_cfg_type, momentum=0.5)
    )

    pooling = StatsPooling(
        emb_dim=configs.model.embed_dim,
        in_plane=arch.dimension
    )
    return nn.Sequential(arch, pooling)
