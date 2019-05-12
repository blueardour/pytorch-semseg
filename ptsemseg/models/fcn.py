
import torch.nn as nn
import torch.nn.functional as F

from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d

# FCN32s
class fcn32s(nn.Module):
    def __init__(self, stem, n_classes=21, learned_billinear=False, **kwargs):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.stem = stem(n_classes)
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.stem.conv_block1(x)
        conv2 = self.stem.conv_block2(conv1)
        conv3 = self.stem.conv_block3(conv2)
        conv4 = self.stem.conv_block4(conv3)
        conv5 = self.stem.conv_block5(conv4)
        score = self.stem.classifier(conv5)
        out = F.interpolate(score, x.size()[2:])
        return out

class fcn16s(nn.Module):
    def __init__(self, stem, n_classes=21, learned_billinear=False, **kwargs):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.stem = stem(args)
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.stem.conv_block1(x)
        conv2 = self.stem.conv_block2(conv1)
        conv3 = self.stem.conv_block3(conv2)
        conv4 = self.stem.conv_block4(conv3)
        conv5 = self.stem.conv_block5(conv4)
        score = self.stem.classifier(conv5)

        score_pool4 = self.score_pool4(conv4)
        score = F.interpolate(score, score_pool4.size()[2:])
        score += score_pool4
        out = F.interpolate(score, x.size()[2:])
        return out

# FCN 8s
class fcn8s(nn.Module):
    def __init__(self, stem, n_classes=21, learned_billinear=False, **kwargs):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.stem = stem(args)

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore4 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore8 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 16, stride=8, bias=False
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )

    def forward(self, x):
        conv1 = self.stem.conv_block1(x)
        conv2 = self.stem.conv_block2(conv1)
        conv3 = self.stem.conv_block3(conv2)
        conv4 = self.stem.conv_block4(conv3)
        conv5 = self.stem.conv_block5(conv4)
        score = self.stem.classifier(conv5)

        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
            ]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[
                :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
            ]

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
            ]
            return out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.interpolate(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.interpolate(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.interpolate(score, x.size()[2:])

        return out

