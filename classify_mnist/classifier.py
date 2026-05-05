"""LogisticRegression 分类器模块。

多项逻辑回归（Softmax 回归），适合与 PCA 降维配合使用。
"""

import numpy as np


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """将标签向量转换为 one-hot 矩阵。

    Args:
        y: 标签，形状 (n_samples,)。
        n_classes: 类别总数。

    Returns:
        One-hot 矩阵，形状 (n_samples, n_classes)。
    """
    one_hot = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


class LogisticRegression:
    """Softmax 回归（多项逻辑回归）分类器。

    使用 L2 正则化的交叉熵损失，mini-batch 梯度下降优化。
    适合与 PCA 降维配合使用（输入为特征向量）。

    Attributes:
        W_: 权重矩阵，形状 (n_features, n_classes)。
        b_: 偏置向量，形状 (n_classes,)。
        n_classes_: 类别数。
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 200,
        batch_size: int = 256,
        reg_strength: float = 1e-4,
    ):
        """初始化 Softmax 回归分类器。

        Args:
            learning_rate: 梯度下降学习率。
            max_iter: 最大迭代轮数（epoch）。
            batch_size: 每批样本数。
            reg_strength: L2 正则化系数。
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.reg_strength = reg_strength
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """在数据 X, y 上训练 Softmax 回归。

        Args:
            X: 训练数据，形状 (n_samples, n_features)。
            y: 训练标签，形状 (n_samples,)。
        """
        n_samples, n_features = X.shape
        self.n_classes_ = int(np.max(y) + 1)

        rng = np.random.default_rng(42)
        scale = np.sqrt(2.0 / n_features)

        # 使用本地变量避免 Optional 类型窄化问题
        W: np.ndarray = rng.normal(0, scale, size=(n_features, self.n_classes_)).astype(
            np.float64
        )
        b: np.ndarray = np.zeros(self.n_classes_, dtype=np.float64)
        self.W_ = W
        self.b_ = b

        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * self.batch_size
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = X_batch @ W + b
                logits -= np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                y_onehot = _one_hot(y_batch, self.n_classes_)
                grad = (probs - y_onehot) / len(y_batch)
                dW = X_batch.T @ grad + self.reg_strength * W
                db = np.sum(grad, axis=0)

                W -= self.learning_rate * dW
                b -= self.learning_rate * db

        self.W_ = W
        self.b_ = b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新样本的类别。

        Args:
            X: 待预测数据，形状 (n_samples, n_features)。

        Returns:
            预测标签，形状 (n_samples,)。
        """
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit 方法。")
        logits = X @ self.W_ + self.b_
        return np.argmax(logits, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算分类准确率。

        Args:
            X: 测试数据，形状 (n_samples, n_features)。
            y: 真实标签，形状 (n_samples,)。

        Returns:
            准确率。
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
