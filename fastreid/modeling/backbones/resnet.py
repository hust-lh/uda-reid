import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., camera=0, drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), sie_xishu=1.0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, camera_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0].reshape(x.shape[0], -1, 1, 1)


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


@BACKBONE_REGISTRY.register()
def build_vit_backbone(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off
    input_size      = cfg.INPUT.SIZE_TRAIN
    pretrain        = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path   = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth           = cfg.MODEL.BACKBONE.DEPTH
    sie_xishu       = cfg.MODEL.BACKBONE.SIE_COE
    stride_size     = cfg.MODEL.BACKBONE.STRIDE_SIZE
    drop_ratio      = cfg.MODEL.BACKBONE.DROP_RATIO
    drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
    attn_drop_rate  = cfg.MODEL.BACKBONE.ATT_DROP_RATE
    # fmt: on

    num_depth = {
        'small': 8,
        'base': 12,
    }[depth]

    num_heads = {
        'small': 8,
        'base': 12,
    }[depth]

    mlp_ratio = {
        'small': 3.,
        'base': 4.
    }[depth]

    qkv_bias = {
        'small': False,
        'base': True
    }[depth]

    qk_scale = {
        'small': 768 ** -0.5,
        'base': None,
    }[depth]

    model = VisionTransformer(img_size=input_size, sie_xishu=sie_xishu, stride_size=stride_size, depth=num_depth,
                              num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop_path_rate=drop_path_ratio, drop_rate=drop_ratio, attn_drop_rate=attn_drop_rate)

    if pretrain:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = model.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed(v, model.pos_embed.data, model.patch_embed.num_y, model.patch_embed.num_x)
                state_dict[k] = v
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
# # encoding: utf-8
# """
# @author:  liaoxingyu
# @contact: sherlockliao01@gmail.com
# """

# import logging
# import math

# import torch
# from torch import nn

# from fastreid.layers import (
#     IBN,
#     Non_local,
#     get_norm,
# )
# from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# from .build import BACKBONE_REGISTRY
# from fastreid.utils import comm


# logger = logging.getLogger(__name__)
# model_urls = {
#     '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
#     'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
#     'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
#     'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
#     'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
# }


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
#                  stride=1, downsample=None, reduction=16):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         if with_ibn:
#             self.bn1 = IBN(planes, bn_norm)
#         else:
#             self.bn1 = get_norm(bn_norm, planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = get_norm(bn_norm, planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
#                  stride=1, downsample=None, reduction=16):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         if with_ibn:
#             self.bn1 = IBN(planes, bn_norm)
#         else:
#             self.bn1 = get_norm(bn_norm, planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = get_norm(bn_norm, planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = get_norm(bn_norm, planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):
#     def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers):
#         self.inplanes = 64
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = get_norm(bn_norm, 64)
#         self.relu = nn.ReLU(inplace=True)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
#         self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
#         self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
#         self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
#         self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

#         self.random_init()

#         # fmt: off
#         if with_nl: self._build_nonlocal(layers, non_layers, bn_norm)
#         else:       self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
#         # fmt: on

#     def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 get_norm(bn_norm, planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

#         return nn.Sequential(*layers)

#     def _build_nonlocal(self, layers, non_layers, bn_norm):
#         self.NL_1 = nn.ModuleList(
#             [Non_local(256, bn_norm) for _ in range(non_layers[0])])
#         self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
#         self.NL_2 = nn.ModuleList(
#             [Non_local(512, bn_norm) for _ in range(non_layers[1])])
#         self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
#         self.NL_3 = nn.ModuleList(
#             [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
#         self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
#         self.NL_4 = nn.ModuleList(
#             [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
#         self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         # layer 1
#         NL1_counter = 0
#         if len(self.NL_1_idx) == 0:
#             self.NL_1_idx = [-1]
#         for i in range(len(self.layer1)):
#             x = self.layer1[i](x)
#             if i == self.NL_1_idx[NL1_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_1[NL1_counter](x)
#                 NL1_counter += 1
#         # layer 2
#         NL2_counter = 0
#         if len(self.NL_2_idx) == 0:
#             self.NL_2_idx = [-1]
#         for i in range(len(self.layer2)):
#             x = self.layer2[i](x)
#             if i == self.NL_2_idx[NL2_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_2[NL2_counter](x)
#                 NL2_counter += 1

#         # layer 3
#         NL3_counter = 0
#         if len(self.NL_3_idx) == 0:
#             self.NL_3_idx = [-1]
#         for i in range(len(self.layer3)):
#             x = self.layer3[i](x)
#             if i == self.NL_3_idx[NL3_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_3[NL3_counter](x)
#                 NL3_counter += 1

#         # layer 4
#         NL4_counter = 0
#         if len(self.NL_4_idx) == 0:
#             self.NL_4_idx = [-1]
#         for i in range(len(self.layer4)):
#             x = self.layer4[i](x)
#             if i == self.NL_4_idx[NL4_counter]:
#                 _, C, H, W = x.shape
#                 x = self.NL_4[NL4_counter](x)
#                 NL4_counter += 1

#         return x

#     def random_init(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)


# def init_pretrained_weights(key):
#     """Initializes model with pretrained weights.

#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     import os
#     import errno
#     import gdown

#     def _get_torch_home():
#         ENV_TORCH_HOME = 'TORCH_HOME'
#         ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
#         DEFAULT_CACHE_DIR = '~/.cache'
#         torch_home = os.path.expanduser(
#             os.getenv(
#                 ENV_TORCH_HOME,
#                 os.path.join(
#                     os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
#                 )
#             )
#         )
#         return torch_home

#     torch_home = _get_torch_home()
#     model_dir = os.path.join(torch_home, 'checkpoints')
#     try:
#         os.makedirs(model_dir)
#     except OSError as e:
#         if e.errno == errno.EEXIST:
#             # Directory already exists, ignore.
#             pass
#         else:
#             # Unexpected OSError, re-raise.
#             raise

#     filename = model_urls[key].split('/')[-1]

#     cached_file = os.path.join(model_dir, filename)

#     if not os.path.exists(cached_file):
#         if comm.is_main_process():
#             gdown.download(model_urls[key], cached_file, quiet=False)

#     comm.synchronize()

#     logger.info(f"Loading pretrained model from {cached_file}")
#     state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

#     return state_dict


# @BACKBONE_REGISTRY.register()
# def build_resnet_backbone(cfg):
#     """
#     Create a ResNet instance from config.
#     Returns:
#         ResNet: a :class:`ResNet` instance.
#     """

#     # fmt: off
#     pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
#     pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
#     last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
#     bn_norm       = cfg.MODEL.BACKBONE.NORM
#     with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
#     with_se       = cfg.MODEL.BACKBONE.WITH_SE
#     with_nl       = cfg.MODEL.BACKBONE.WITH_NL
#     depth         = cfg.MODEL.BACKBONE.DEPTH
#     # fmt: on

#     num_blocks_per_stage = {
#         '18x': [2, 2, 2, 2],
#         '34x': [3, 4, 6, 3],
#         '50x': [3, 4, 6, 3],
#         '101x': [3, 4, 23, 3],
#     }[depth]

#     nl_layers_per_stage = {
#         '18x': [0, 0, 0, 0],
#         '34x': [0, 0, 0, 0],
#         '50x': [0, 2, 3, 0],
#         '101x': [0, 2, 9, 0]
#     }[depth]

#     block = {
#         '18x': BasicBlock,
#         '34x': BasicBlock,
#         '50x': Bottleneck,
#         '101x': Bottleneck
#     }[depth]

#     model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
#                    num_blocks_per_stage, nl_layers_per_stage)

#     if pretrain:
#         # Load pretrain path if specifically
#         if pretrain_path:
#             try:
#                 state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
#                 logger.info(f"Loading pretrained model from {pretrain_path}")
#             except FileNotFoundError as e:
#                 logger.info(f'{pretrain_path} is not found! Please check this path.')
#                 raise e
#             except KeyError as e:
#                 logger.info("State dict keys error! Please check the state dict.")
#                 raise e
#         else:
#             key = depth
#             if with_ibn: key = 'ibn_' + key
#             if with_se:  key = 'se_' + key

#             state_dict = init_pretrained_weights(key)

#         incompatible = model.load_state_dict(state_dict, strict=False)
#         if incompatible.missing_keys:
#             logger.info(
#                 get_missing_parameters_message(incompatible.missing_keys)
#             )
#         if incompatible.unexpected_keys:
#             logger.info(
#                 get_unexpected_parameters_message(incompatible.unexpected_keys)
#             )

#     return model
