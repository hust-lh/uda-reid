# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn
import torch
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
import collections
import random
class SmoothingForImage(object):
    def __init__(self, momentum=0.1, num=1):

        self.map = dict()
        self.momentum = momentum
        self.num = num


    def get_soft_label(self, path, feature):

        #feature = torch.cat(feature, dim=1)
        soft_label = []

        for j,p in enumerate(path):

            current_soft_feat = feature[j*self.num:(j+1)*self.num, :].clone().mean(dim=0)
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()

            key  = p
            if key not in self.map:
                self.map.setdefault(key, current_soft_feat)
                soft_label.append(self.map[key])
            else:
                self.map[key] = self.map[key]*(1-self.momentum) + self.momentum*current_soft_feat
                soft_label.append(self.map[key])
        soft_label = torch.stack(soft_label, dim=0).cuda()                              #通常为了保留–[序列(先后)信息] 和 [张量的矩阵信息] 才会使用stack
        return soft_label


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """ __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    in_planes = 2048

    def __init__(self, num_pids=None, last_stride=1):
        super(ResNetBuilder, self).__init__()
        depth = 152

        final_layer = 'layer4'
        self.final_layer = final_layer

        pretrained = True

        # Construct base (pretrained) resnet
        if depth not in ResNetBuilder.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNetBuilder.__factory[depth](pretrained=pretrained)
        if depth < 50:
            out_planes = fea_dims_small[final_layer]
        else:
            out_planes = fea_dims[final_layer]

        i = 0
        for module in self.base.modules():
            #print(module)
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.momentum = None
                #print(module.momentum)
                i+=1
        print(i) """

    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        num_pids      = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        norm_type     = cfg.MODEL.HEADS.NORM
        self.num_cameras = cfg.DATASETS.NUM_CAMERAS

    # def __init__(self, num_pids=None, last_stride=1):
    #     super().__init__() 
        self.num_pids = num_pids
        # print(num_pids)
        # self.base = ResNet_Backbone(last_stride)
        # model_path = '/home/' + getuser() + '/.cache/torch/checkpoints/resnet50-19c8e357.pth'
        # self.base.load_param(model_path)
        # self.fc = nn.Linear(2048, 2048) 
        bn_neck = nn.BatchNorm1d(768,  eps=1e-05, momentum=0.1,)
        bn_neck.bias.requires_grad_(False)
        self.bottleneck = nn.Sequential(bn_neck)
        self.bottleneck.apply(weights_init_kaiming)
        # self.pooling = GeneralizedMeanPoolingP()
        
        embedding_size = 768
        self.RFM = RFM(embedding_size)                                              #    Camera feature extrctor "E" in paper.
        if self.num_pids is not None:
            self.classifier = nn.Linear(768, self.num_pids, bias=False)             #   人id分类器 
            self.classifier.apply(weights_init_classifier)

            #camera embedding and classfier
            
            self.DAL = DAL_regularizer(embedding_size)                              #      判别器


            if self.num_pids == 4101:
                cam_num = 15
            elif self.num_pids == 751:
                cam_num = 6
            else:
                cam_num = 8
            self.cam_classifier = nn.Sequential(nn.Linear(embedding_size, embedding_size, bias=False) \
                                                , nn.BatchNorm1d(embedding_size)
                                                , nn.ReLU(inplace=True)
                                                ,nn.Linear(embedding_size, cam_num, bias=False))        #camera_id分类器
            self.cam_classifier.apply(weights_init_classifier)


            #momentum updating
            self.agent = SmoothingForImage(momentum=0.9, num=1)            
            
    def forward(self, x, path=None):

        # x = self.base(x)
        feat_before_bn = x

        feat_before_bn = F.avg_pool2d(feat_before_bn, feat_before_bn.shape[2:])
        feat_before_bn = feat_before_bn.view(feat_before_bn.shape[0], -1)
        # feat_before_bn = self.fc(feat_before_bn)
        # print(feat_before_bn.size())
        # feat_before_bn = feat_before_bn[...,0,0]
        feat_after_bn = self.bottleneck(feat_before_bn)
        detach_fea = feat_before_bn.detach()

        if path:
            detach_fea = self.agent.get_soft_label(path, detach_fea)

        detach_fea.requires_grad_()
        detach_fea = feat_before_bn

        latent_code = self.RFM(detach_fea)
        pzx = torch.cat((feat_before_bn, latent_code), 1 )
        random_index = random.sample(range(0,latent_code.size(0)),latent_code.size(0))

        random_code = latent_code[random_index]
        pzpx = torch.cat((feat_before_bn, random_code), 1 )

        
        


        # Evaluation
        # fmt: off
        if not self.training: return feat_after_bn
        # fmt: on
        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            classification_results = self.classifier(feat_after_bn)
            pred_class_logits = F.linear(feat_after_bn, self.classifier.weight)
            
            classification_results = self.classifier(latent_code)
            pzx_scores, pzpx_scores = self.DAL(pzx, pzpx)

            cam_scores = self.cam_classifier(latent_code)
            # cam_score = self.cam_classifier(feat_after_bn)

            return {
                "features":feat_before_bn,
                "feat_after_bn": feat_after_bn,
                "classification_results": classification_results,
                "pred_class_logits":pred_class_logits,
                "cam_scores": cam_scores,
                'pzx_scores': pzx_scores,
                'pzpx_scores': pzpx_scores
                }
        else:
            return {
                "feat_after_bn": feat_after_bn,}

    def get_optim_policy(self):
        base_param_group = filter(lambda p: p.requires_grad, self.base.parameters())
        add_param_group = filter(lambda p: p.requires_grad, self.bottleneck.parameters())
        cls_param_group = filter(lambda p: p.requires_grad, self.classifier.parameters())
        RFM_param_group = filter(lambda p: p.requires_grad, self.RFM.parameters())
        DAL_param_group = filter(lambda p: p.requires_grad, self.DAL.parameters())
        cam_cls_param_group = filter(lambda p: p.requires_grad, self.cam_classifier.parameters())

        all_param_groups = []
        all_param_groups.append({'params': base_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': add_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cls_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': RFM_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': DAL_param_group, "weight_decay": 0.0005})
        all_param_groups.append({'params': cam_cls_param_group, "weight_decay": 0.0005})        
        return all_param_groups

#camera clssfier
class RFM(nn.Module):
    '''
    Camera feature extrctor "E" in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.seq = nn.Sequential(nn.BatchNorm1d(n_in)
                                , nn.Linear(n_in, n_in)
                                , nn.BatchNorm1d(n_in)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(n_in, n_in)
                                , nn.BatchNorm1d(n_in)
                                , nn.ReLU(inplace=True))
    
    def forward(self, xs):
        return self.seq(xs)

class DAL_regularizer(nn.Module):
    '''
    Disentangled Feature Learning module in paper.
    '''
    def __init__(self, n_in):
        super().__init__()
        self.discrimintor = nn.Sequential(nn.Linear(n_in*2, n_in)
                                         , nn.ReLU(inplace=True)
                                         , nn.Linear(n_in, n_in)
                                         , nn.ReLU(inplace=True)
                                         , nn.Linear(n_in, 1)
                                         , nn.Sigmoid())
        
    
    def forward(self, pzx, pzpx):
        pzx_scores = self.discrimintor(pzx)
        pzpx_scores = self.discrimintor(pzpx)

        return pzx_scores, pzpx_scores

# class EmbeddingHead(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # fmt: off
#         feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
#         num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
#         neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
#         pool_type     = cfg.MODEL.HEADS.POOL_LAYER
#         cls_type      = cfg.MODEL.HEADS.CLS_LAYER
#         norm_type     = cfg.MODEL.HEADS.NORM
#         self.num_cameras = cfg.DATASETS.NUM_CAMERAS

#         if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
#         elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
#         elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
#         elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
#         else:                            raise KeyError(f"{pool_type} is not supported!")
#         # fmt: on

#         self.neck_feat = neck_feat
#         print(111,norm_type)
#         self.bottleneck = get_norm(norm_type, feat_dim, bias_freeze=True)

#         #camera-specific batch normalization
#         self.bottleneck_camera = nn.ModuleList(torch.nn.BatchNorm2d(2048) for _ in range(self.num_cameras))
#         self.bottleneck_camera_map = nn.ModuleList(torch.nn.BatchNorm2d(2048) for _ in range(self.num_cameras))

#         # classification layer
        
#         # fmt: off
#         if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
#         else:                             raise KeyError(f"{cls_type} is not supported!")
#         # fmt: on

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
 
#         for bn in self.bottleneck_camera:
#             bn.apply(weights_init_kaiming)
#         for bn in self.bottleneck_camera_map:
#             bn.apply(weights_init_kaiming)

#     def forward(self, features, targets=None, camids=None):
#         """
#         See :class:`ReIDHeads.forward`.
#         """
#         global_feat = self.pool_layer(features)#全局特征

#         bn_feat = self.bottleneck(global_feat)#全局过bn特征
#         bn_feat = bn_feat[..., 0, 0]

#         feature_after_bn_list = []
#         feature_map_list = []
#         feature_dict_out = collections.defaultdict(list)
#         feature_map_dict_out = collections.defaultdict(list)
#         global_feat_map = self.bottleneck(features)#局部全局特征
#         feature_map_out = []
#         fake_feature_out = []

#         if not self.training:
#             return bn_feat, global_feat_map
#         else:
#             uniq = torch.unique(camids)
#             for c in uniq:
#                 index = torch.where(c == camids)[0]
#                 feature_after_bn_list.append(global_feat[index])
#                 feature_map_list.append(features[index])

#             for cid, feature_cid_per,feature_map in zip(uniq,feature_after_bn_list, feature_map_list):
#                 for i in range(self.num_cameras):
#                     if i == int(cid):
#                         feature_dict_out[int(cid)].append(self.bottleneck_camera[i](feature_cid_per)[..., 0, 0])
#                         feature_map_dict_out[int(cid)].append(self.bottleneck_camera_map[i](feature_map))
#                     else:
#                         self.bottleneck_camera[i].eval()
#                         self.bottleneck_camera_map[i].eval()
#                         feature_dict_out[int(cid)].append(self.bottleneck_camera[i](feature_cid_per)[..., 0, 0])
#                         feature_map_dict_out[int(cid)].append(self.bottleneck_camera_map[i](feature_map))
#                         self.bottleneck_camera[i].train()
#                         self.bottleneck_camera_map[i].train()

#             for i, (key, values) in enumerate(feature_map_dict_out.items()):
#                 if i == 0:
#                     for value in values:
#                         feature_map_out.append(value)
#                 if i == 1:
#                     for j, value in enumerate(values):
#                         feature_map_out[j] = torch.cat((feature_map_out[j], value))

#             for i, (key, values) in enumerate(feature_dict_out.items()):
#                 if i == 0:
#                     for value in values:
#                         fake_feature_out.append(value)
#                 if i == 1:
#                     for j, value in enumerate(values):
#                         fake_feature_out[j] = torch.cat((fake_feature_out[j], value))

#         # Evaluation
#         # fmt: off
#         if not self.training: return bn_feat
#         # fmt: on
#         # Training
#         if self.classifier.__class__.__name__ == 'Linear':
#             cls_outputs = self.classifier(bn_feat)
#             pred_class_logits = F.linear(bn_feat, self.classifier.weight)
#         else:
#             cls_outputs = self.classifier(bn_feat, targets)
#             pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
#                                                              F.normalize(self.classifier.weight))

#         # fmt: off
#         if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
#         elif self.neck_feat == "after": feat = bn_feat
#         else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
#         # fmt: on

#         return {
#             "cls_outputs": cls_outputs,
#             "pred_class_logits": pred_class_logits,
#             "features": feat,
#             'bn_feat': bn_feat,
#             'feature_dict_bn': feature_dict_out,
#             'feature_map_dict': feature_map_dict_out,
#             'feature_map_out': feature_map_out,
#             'global_feature_map': global_feat_map,
#         }
