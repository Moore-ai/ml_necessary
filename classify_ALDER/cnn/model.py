"""CNN 网络结构定义（4 个 Conv-BN-ReLU-MaxPool 块）。

输入 (3, 224, 224) RGB 图像，输出 5 类 logits。
"""

import torch
from torch import nn


class AIDERCNN(nn.Module):
    """AIDER 分类 CNN 网络。

    4 个 Conv-BN-ReLU-MaxPool 块 + AdaptiveAvgPool + 2 层 FC。

    Attributes:
        features: 卷积特征提取部分。
        classifier: 分类头（FC 层）。
    """

    def __init__(self, n_classes: int = 5, dropout: float = 0.5) -> None:
        """初始化 AIDERCNN。

        Args:
            n_classes: 输出类别数。
            dropout: Dropout 概率。
        """
        super().__init__()

        self.features = nn.Sequential(
            # Block 1:  3 → 32,  224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2:  32 → 64,  112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3:  64 → 128,  56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4:  128 → 256,  28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量，形状 (batch_size, 3, 224, 224)。

        Returns:
            输出 logits，形状 (batch_size, n_classes)。
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
