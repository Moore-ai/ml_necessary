"""CNN 分类器模块（基于 PyTorch）。

使用两层卷积 + 全连接网络对 MNIST 手写数字进行分类。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CNN(nn.Module):
    """内部 CNN 网络结构。

    输入: (batch, 1, 28, 28)
    输出: (batch, 10) 的 logits

    结构:
      Conv1(1→32, 3×3) → ReLU → MaxPool(2×2)
      Conv2(32→64, 3×3) → ReLU → MaxPool(2×2)
      FC(1600→128) → ReLU → FC(128→10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))   # 28→26→13
        x = F.relu(F.max_pool2d(self.conv2(x), 2))   # 13→11→5
        x = x.view(x.size(0), -1)                     # → (batch, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNClassifier:
    """CNN 手写数字分类器。

    封装 PyTorch CNN 模型的训练与预测接口。
    支持 GPU 自动切换（CUDA 可用时优先使用 GPU）。

    Attributes:
        batch_size: 训练批大小。
        epochs: 训练轮数。
        lr: 学习率。
    """

    def __init__(
        self,
        batch_size: int = 64,
        epochs: int = 10,
        lr: float = 0.001,
    ):
        """初始化 CNN 分类器。

        Args:
            batch_size: 训练批大小。
            epochs: 训练轮数。
            lr: Adam 优化器学习率。
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self._model: _CNN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """在数据 X, y 上训练 CNN。

        Args:
            X: 图像数据，形状 (n_samples, 1, 28, 28)，float32 类型。
            y: 标签，形状 (n_samples,)，整数类型。
        """
        self._model = _CNN().to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        X_t = torch.from_numpy(X).to(self._device)
        y_t = torch.from_numpy(y).to(self._device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self._model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch == 1 or epoch % 5 == 0:
                avg_loss = total_loss / len(loader)
                print(f"     epoch {epoch:>3d}/{self.epochs}  loss={avg_loss:.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新样本的类别。

        Args:
            X: 图像数据，形状 (n_samples, 1, 28, 28)。

        Returns:
            预测标签，形状 (n_samples,)。
        """
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法。")
        self._model.eval()
        X_t = torch.from_numpy(X).to(self._device)
        logits = self._model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率。

        Args:
            X: 测试数据，形状 (n_samples, 1, 28, 28)。
            y: 真实标签，形状 (n_samples,)。

        Returns:
            准确率。
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
