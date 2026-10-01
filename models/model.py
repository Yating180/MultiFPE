# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DEB.VisualBackbone import VisualBackbone
from models.DEB.TCN import TemporalConvNet
from models.DEB.av_crossatten import DCNLayer
from torch.nn import Linear, BatchNorm1d
from models.DEB.landmark_feature import LandmarkFeatureExtractor, LandmarkFeatureExtractor1
from models.vit_decoder_two import decoder_fuser
from models.SEB.model_StaticModel import StaticModel
import torch.nn.init as init
from models.DEB.dynamic_gateconcat import DynamicWeightedConcat2


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = VisualBackbone(use_pretrained=True, state_dict_path='C:\\Users\\DELL\\Desktop\\MultiFPE\\ckpt\\ir50.pth')
        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.shape
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.model(x)
        features = features.view(batch_size, num_frames, -1)
        return features


class FacialPalsyClassificationModel(nn.Module):
    def __init__(self, output_dim):
        super(FacialPalsyClassificationModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.static_model = StaticModel()
        self.tcn_expression = TemporalConvNet(512, [512, 512, 512, 512], kernel_size=5, dropout=0.1, attention=0)
        self.tcn_landmark = TemporalConvNet(num_inputs=30, num_channels=[64, 64, 32, 32], kernel_size=5, dropout=0.1, attention=0)
        self.bn_expression = BatchNorm1d(512)
        self.bn_landmark = BatchNorm1d(32)
        self.coattn = DCNLayer(32, 512, 16, dropout=0.6)
        self.gateconcat = DynamicWeightedConcat2(feat1_dim=32, feat2_dim=512, hidden_dim=64)
        self.fc1 = Linear(544, 512)
        self.decoder_fuser = decoder_fuser(512, 8, 4, 0.6)
        self.fc2 = Linear(1024, output_dim)

    def forward(self, dynamic_images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor, mask=None):
        device = next(self.parameters()).device

        dynamic_images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor,  = [
            x.to(device) for x in [dynamic_images, static_image, landmarks_tensor, left_iris_tensor, right_iris_tensor, static_landmarks_tensor, static_left_iris_tensor, static_right_iris_tensor]
        ]

        expression_features = self.feature_extractor(dynamic_images).transpose(1, 2)
        expression_features = self.tcn_expression(expression_features)
        expression_features = self.bn_expression(expression_features).transpose(1, 2)

        landmark_features0 = LandmarkFeatureExtractor.extract_features(landmarks_tensor, left_iris_tensor,
                                                                       right_iris_tensor)
        landmark_features1 = LandmarkFeatureExtractor1.extract_features(landmarks_tensor, left_iris_tensor,
                                                                        right_iris_tensor)
        landmark_features = torch.cat([landmark_features0, landmark_features1], dim=-1)
        landmark_features = self.tcn_landmark(landmark_features.transpose(1, 2))
        landmark_features = self.bn_landmark(landmark_features).transpose(1, 2)

        landmarkfeature, imagefeature = self.coattn(landmark_features, expression_features)
        d = self.gateconcat(landmarkfeature, imagefeature)
        d = d[:, -1, :]
        d = self.fc1(d)

        s = self.static_model(static_image)

        d, s = self.decoder_fuser(d, s)

        output = torch.cat([d, s], dim=-1)

        output = self.fc2(output)

        return output
