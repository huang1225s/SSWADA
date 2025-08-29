import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ResClassifier(nn.Module):
    def __init__(self, class_num, extract=False, training=False, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc2 = nn.Linear(256, class_num)
        self.extract = extract
        self.training = training
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1 - self.dropout_p))
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit
        return logit

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=6):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class GC(nn.Module):

    def __init__(self, inplans, planes, stride=1):
        super(GC, self).__init__()
        self.conv = conv(inplans, planes, kernel_size=9, padding=9//2,
                            stride=stride, groups=16)
        self.se = SEWeightModule(planes)
        self.split_channel = planes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)

        feats = x
        feats = feats.view(batch_size, 1, self.split_channel, feats.shape[2], feats.shape[3])

        x_se = self.se(x)
        x_se = x_se
        attention_vectors = x_se.view(batch_size, 1, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(1):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

class Features(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, patch_size, n_planes=2):
        super(Features, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        self.GC = GC(self.input_channels, self.input_channels)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, 2048)
        # self.drop = nn.Dropout(0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x1 = self.GC(x)
        x = x1 + x
        x2 = x.unsqueeze(1)
        x2 = F.relu(self.conv1(x2))
        x = F.relu(self.conv2(x2))

        x = x.view(-1, self.features_size)
        # x = self.drop(x)
        x = self.fc(x)
        return x