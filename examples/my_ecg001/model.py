import torch
import torch.nn as nn


class TinyECGCNN(nn.Module):
    """
    最稳最轻：
    Conv(1->4) + ReLU + GAP + FC
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(4, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyECGCNN8(nn.Module):
    """
    通道数提升版：
    Conv(1->8) + ReLU + GAP + FC
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(8, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyECGTwoConv(nn.Module):
    """
    两层卷积但只保留一个 ReLU：
    Conv(1->4) + Conv(4->8) + ReLU + GAP + FC
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(8, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyECGMLPHead(nn.Module):
    """
    在分类头增加一层，但会多一个 ReLU。
    baseline 可试，后续上 FHE 时要谨慎。
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(8, 8, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(8, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=2, model_name='tiny_cnn'):
    model_name = model_name.lower()

    if model_name == 'tiny_cnn':
        return TinyECGCNN(num_classes=num_classes)
    if model_name == 'tiny_cnn8':
        return TinyECGCNN8(num_classes=num_classes)
    if model_name == 'two_conv':
        return TinyECGTwoConv(num_classes=num_classes)
    if model_name == 'mlp_head':
        return TinyECGMLPHead(num_classes=num_classes)

    raise ValueError(f'不支持的 model_name: {model_name}')


if __name__ == '__main__':
    for name in ['tiny_cnn', 'tiny_cnn8', 'two_conv', 'mlp_head']:
        model = build_model(num_classes=2, model_name=name)
        x = torch.randn(2, 1, 16, 16)
        y = model(x)
        print(name, y.shape)