# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers  #3
        h = [hidden_dim] * (num_layers - 1) #[256,256]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        #256*256 256*256 256*4

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    return Transformer(
        d_model=cfg.MODEL.Transformer.hidden_dim,
        dropout=cfg.MODEL.Transformer.dropout,
        nhead=cfg.MODEL.Transformer.nheads,
        dim_feedforward=cfg.MODEL.Transformer.dim_feedforward,
        num_encoder_layers=cfg.MODEL.Transformer.enc_layers,
        num_decoder_layers=cfg.MODEL.Transformer.dec_layers,
        normalize_before=cfg.MODEL.Transformer.pre_norm,
        return_intermediate_dec=False,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# # ------------------------------------------------------------------------
# # Deformable DETR
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------

# # from models.ops.modules import MSDeformAttn
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
# import copy
# from typing import Optional, List
# import math

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


# import warnings
# import math

# # from ..functions import MSDeformAttnFunction

# from torch.autograd import Function
# from torch.autograd.function import once_differentiable

# import MultiScaleDeformableAttention as MSDA


# class MSDeformAttnFunction(Function):
#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         ctx.im2col_step = im2col_step
#         output = MSDA.ms_deform_attn_forward(
#             value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value, grad_sampling_loc, grad_attn_weight = \
#             MSDA.ms_deform_attn_backward(
#                 value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

#         return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
# def _is_power_of_2(n):
#     if (not isinstance(n, int)) or (n < 0):
#         raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
#     return (n & (n-1) == 0) and n != 0
# class MSDeformAttn(nn.Module):
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
#         """
#         Multi-Scale Deformable Attention Module
#         :param d_model      hidden dimension
#         :param n_levels     number of feature levels
#         :param n_heads      number of attention heads
#         :param n_points     number of sampling points per attention head per feature level
#         """
#         super().__init__()
#         if d_model % n_heads != 0:
#             raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
#         _d_per_head = d_model // n_heads
#         # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
#         if not _is_power_of_2(_d_per_head):
#             warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
#                           "which is more efficient in our CUDA implementation.")

#         self.im2col_step = 64

#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_heads = n_heads
#         self.n_points = n_points

#         self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
#         self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
#         self.value_proj = nn.Linear(d_model, d_model)
#         self.output_proj = nn.Linear(d_model, d_model)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         constant_(self.sampling_offsets.weight.data, 0.)
#         thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
#         for i in range(self.n_points):
#             grid_init[:, :, i, :] *= i + 1
#         with torch.no_grad():
#             self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
#         constant_(self.attention_weights.weight.data, 0.)
#         constant_(self.attention_weights.bias.data, 0.)
#         xavier_uniform_(self.value_proj.weight.data)
#         constant_(self.value_proj.bias.data, 0.)
#         xavier_uniform_(self.output_proj.weight.data)
#         constant_(self.output_proj.bias.data, 0.)

#     def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
#         """
#         :param query                       (N, Length_{query}, C)
#         :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
#                                         or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
#         :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
#         :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#         :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
#         :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
#         :return output                     (N, Length_{query}, C)
#         """
#         N, Len_q, _ = query.shape
#         N, Len_in, _ = input_flatten.shape
#         assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

#         value = self.value_proj(input_flatten)
#         if input_padding_mask is not None:
#             value = value.masked_fill(input_padding_mask[..., None], float(0))
#         value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
#         sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
#         attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
#         attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
#         # N, Len_q, n_heads, n_levels, n_points, 2
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                                  + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] \
#                                  + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
#         else:
#             raise ValueError(
#                 'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
#         output = MSDeformAttnFunction.apply(
#             value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
#         output = self.output_proj(output)
#         return output
    
# def inverse_sigmoid(x, eps=1e-5):
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1/x2)
# class DeformableTransformer(nn.Module):
#     def __init__(self, d_model=256, nhead=8,
#                  num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
#                  activation="relu", return_intermediate_dec=False,
#                  num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
#                  two_stage=False, two_stage_num_proposals=300):
#         super().__init__()

#         self.d_model = d_model
#         self.nhead = nhead
#         self.two_stage = two_stage
#         self.two_stage_num_proposals = two_stage_num_proposals

#         encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, enc_n_points)
#         self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

#         decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, dec_n_points)
#         self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

#         self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

#         if two_stage:
#             self.enc_output = nn.Linear(d_model, d_model)
#             self.enc_output_norm = nn.LayerNorm(d_model)
#             self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
#             self.pos_trans_norm = nn.LayerNorm(d_model * 2)
#         else:
#             self.reference_points = nn.Linear(d_model, 2)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         if not self.two_stage:
#             xavier_uniform_(self.reference_points.weight.data, gain=1.0)
#             constant_(self.reference_points.bias.data, 0.)
#         normal_(self.level_embed)

#     def get_proposal_pos_embed(self, proposals):
#         num_pos_feats = 128
#         temperature = 10000
#         scale = 2 * math.pi

#         dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
#         dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
#         # N, L, 4
#         proposals = proposals.sigmoid() * scale
#         # N, L, 4, 128
#         pos = proposals[:, :, :, None] / dim_t
#         # N, L, 4, 64, 2
#         pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
#         return pos

#     def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
#         N_, S_, C_ = memory.shape
#         base_scale = 4.0
#         proposals = []
#         _cur = 0
#         for lvl, (H_, W_) in enumerate(spatial_shapes):
#             mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
#             valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#             valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

#             grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
#                                             torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
#             grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

#             scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
#             grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
#             wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
#             proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
#             proposals.append(proposal)
#             _cur += (H_ * W_)
#         output_proposals = torch.cat(proposals, 1)
#         output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
#         output_proposals = torch.log(output_proposals / (1 - output_proposals))
#         output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
#         output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

#         output_memory = memory
#         output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
#         output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
#         output_memory = self.enc_output_norm(self.enc_output(output_memory))
#         return output_memory, output_proposals

#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio

#     def forward(self, srcs, masks, pos_embeds, query_embed=None):
#         assert self.two_stage or query_embed is not None

#         # prepare input for encoder
#         src_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
#             bs, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             src = src.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             src_flatten.append(src)
#             mask_flatten.append(mask)
#         src_flatten = torch.cat(src_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

#         # encoder
#         memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

#         # prepare input for decoder
#         bs, _, c = memory.shape
#         if self.two_stage:
#             output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

#             # hack implementation for two-stage Deformable DETR
#             enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
#             enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

#             topk = self.two_stage_num_proposals
#             topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
#             topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
#             topk_coords_unact = topk_coords_unact.detach()
#             reference_points = topk_coords_unact.sigmoid()
#             init_reference_out = reference_points
#             pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
#             query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
#         else:
#             query_embed, tgt = torch.split(query_embed, c, dim=1)
#             query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
#             tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
#             reference_points = self.reference_points(query_embed).sigmoid()
#             init_reference_out = reference_points

#         # decoder
#         hs, inter_references = self.decoder(tgt, reference_points, memory,
#                                             spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

#         inter_references_out = inter_references
#         if self.two_stage:
#             return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
#         return hs, init_reference_out, inter_references_out, None, None


# class DeformableTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4):
#         super().__init__()

#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
#         # self attention
#         src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         return src


# class DeformableTransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers

#     @staticmethod
#     def get_reference_points(spatial_shapes, valid_ratios, device):
#         reference_points_list = []
#         for lvl, (H_, W_) in enumerate(spatial_shapes):

#             ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#                                           torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points

#     def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
#         output = src
#         reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
#         for _, layer in enumerate(self.layers):
#             output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

#         return output


# class DeformableTransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4):
#         super().__init__()

#         # cross attention
#         self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
#         # self attention
#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         # cross attention
#         tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
#                                reference_points,
#                                src, src_spatial_shapes, level_start_index, src_padding_mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt


# class DeformableTransformerDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.return_intermediate = return_intermediate
#         # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
#         self.bbox_embed = None
#         self.class_embed = None

#     def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
#                 query_pos=None, src_padding_mask=None):
#         output = tgt

#         intermediate = []
#         intermediate_reference_points = []
#         for lid, layer in enumerate(self.layers):
#             if reference_points.shape[-1] == 4:
#                 reference_points_input = reference_points[:, :, None] \
#                                          * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
#             else:
#                 assert reference_points.shape[-1] == 2
#                 reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
#             output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

#             # hack implementation for iterative bounding box refinement
#             if self.bbox_embed is not None:
#                 tmp = self.bbox_embed[lid](output)
#                 if reference_points.shape[-1] == 4:
#                     new_reference_points = tmp + inverse_sigmoid(reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 else:
#                     assert reference_points.shape[-1] == 2
#                     new_reference_points = tmp
#                     new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
#                     new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()

#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)

#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(intermediate_reference_points)

#         return output, reference_points


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# def build_deforamble_transformer(cfg):
#     return DeformableTransformer(
#         d_model=cfg.MODEL.Transformer.hidden_dim,
#         nhead=cfg.MODEL.Transformer.nheads,
#         num_encoder_layers=cfg.MODEL.Transformer.enc_layers,
#         num_decoder_layers=cfg.MODEL.Transformer.dec_layers,
#         dim_feedforward=cfg.MODEL.Transformer.dim_feedforward,
#         dropout=cfg.MODEL.Transformer.dropout,
#         activation="relu",
#         return_intermediate_dec=True,
#         num_feature_levels=cfg.MODEL.Transformer.num_feature_levels,
#         dec_n_points=cfg.MODEL.Transformer.dec_n_points,
#         enc_n_points=cfg.MODEL.Transformer.enc_n_points,
#         two_stage=cfg.MODEL.Transformer.two_stage,
#         two_stage_num_proposals=cfg.MODEL.Transformer.num_queries)


