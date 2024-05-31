import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(output_channel)

        self.conv2 = nn.Conv2d(
            output_channel, output_channel, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(output_channel)

        self.conv3 = nn.Conv2d(
            output_channel, output_channel * self.expansion, kernel_size=1, padding=0
        )
        self.batch_norm3 = nn.BatchNorm2d(output_channel * self.expansion)

        self.stride = stride
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        Block: nn.Module,
        layers: list = [3, 4, 6, 3],
        num_classes: int = 2,
        num_channels: int = 3,
    ):
        """
        ResNet50
        :param Block: BottleBlock
        :param layers: [3,4,6,3]
        :param num_classes: number of classes
        :param num_channels: number of channels
        """
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Block, layers[0], 64, 1)
        self.layer2 = self._make_layer(Block, layers[1], 128, 2)
        self.layer3 = self._make_layer(Block, layers[2], 256, 2)
        self.layer4 = self._make_layer(Block, layers[3], 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

    def _make_layer(self, Block: nn.Module, num_blocks: int, channel: int, stride: int):
        downsample = None
        layers = []

        if stride != 1 or self.in_channel != channel * Block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * Block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channel * Block.expansion),
            )

        layers.append(Block(self.in_channel, channel, stride, downsample))
        self.in_channel = channel * Block.expansion
        for _ in range(1, num_blocks):
            layers.append(Block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet50(num_classes: int = 6, num_channels: int = 3):
    return ResNet(BottleBlock, [3, 4, 6, 3], num_classes, num_channels)
