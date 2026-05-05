"""PCA（主成分分析）实现模块。

使用协方差矩阵特征分解实现 PCA，支持数据降维。
当 n_features <= n_samples 时使用协方差矩阵分解，效率更高。
"""

import numpy as np


class PCA:
    """主成分分析（Principal Component Analysis）。

    使用协方差矩阵特征分解计算主成分，
    适用于 n_features <= n_samples 的场景。

    Attributes:
        n_components: 保留的主成分数量。
        mean_: 训练数据的均值向量。
        components_: 主成分方向矩阵，形状 (n_components, n_features)。
        singular_values_: 奇异值。
        explained_variance_ratio_: 各主成分解释的方差比例。
    """

    def __init__(self, n_components: int | None = None):
        """初始化 PCA。

        Args:
            n_components: 保留的主成分数量。
                          若为 None，则保留所有主成分。
        """
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """在数据 X 上拟合 PCA 模型。

        使用协方差矩阵特征分解（eigh），
        对 n_features=784 的 MNIST 数据比 SVD 快数十倍。

        Args:
            X: 训练数据，形状 (n_samples, n_features)。

        Returns:
            已拟合的 PCA 实例。
        """
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)

        n_comp = self.n_components if self.n_components is not None else max_components
        if n_comp > max_components:
            n_comp = max_components

        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 协方差矩阵: X^T X / (n-1)，形状 (n_features, n_features)
        # 对 MNIST (60000, 784) 只需计算 784×784 矩阵
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # 特征分解（eigh 将特征值按升序排列）
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 翻转为降序
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # 取前 n_comp 个
        self.components_ = eigenvectors[:, :n_comp].T
        self.explained_variance_ratio_ = eigenvalues[:n_comp] / np.sum(eigenvalues)
        self.singular_values_ = np.sqrt(eigenvalues[:n_comp] * (n_samples - 1))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """将数据投影到主成分空间。

        Args:
            X: 数据，形状 (n_samples, n_features)。

        Returns:
            降维后的数据，形状 (n_samples, n_components)。
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA 模型尚未拟合，请先调用 fit 方法。")
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在数据 X 上拟合 PCA 并同时转换。

        Args:
            X: 数据，形状 (n_samples, n_features)。

        Returns:
            降维后的数据，形状 (n_samples, n_components)。
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """将降维后的数据映射回原始空间。

        Args:
            X: 降维数据，形状 (n_samples, n_components)。

        Returns:
            重建数据，形状 (n_samples, n_features)。
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA 模型尚未拟合，请先调用 fit 方法。")
        return X @ self.components_ + self.mean_
