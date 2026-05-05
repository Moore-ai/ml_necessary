"""MNIST PCA 降维对比实验主脚本。

对比不同 PCA 主成分数量下 LogisticRegression 的分类效果。
"""

import time
from typing import List, Tuple

import numpy as np

from classify_mnist.classifier import LogisticRegression
from classify_mnist.data_loader import load_mnist
from classify_mnist.pca import PCA

_N_TRAIN = 12000           # 训练集样本数
_N_TEST = 2000             # 测试集样本数
_N_COMPONENTS: List[int] = [5, 10, 20, 50, 100, 200, 392]  # PCA 主成分数


def _run_pca_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int,
) -> Tuple[float, float, float]:
    """使用 PCA 降维后训练 LogisticRegression。

    Args:
        X_train: 平坦训练数据 (n_train, 784)。
        y_train: 训练标签。
        X_test: 平坦测试数据 (n_test, 784)。
        y_test: 测试标签。
        n_components: PCA 主成分数。

    Returns:
        (准确率, PCA 用时, 训练用时)
    """
    t0 = time.perf_counter()
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    t_pca = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = LogisticRegression(learning_rate=0.1, max_iter=200, batch_size=256)
    clf.fit(X_train_pca, y_train)
    t_train = time.perf_counter() - t0

    acc = clf.score(X_test_pca, y_test)
    return acc, t_pca, t_train


def main() -> None:
    """主流程: 加载数据 -> PCA 降维 -> LR 分类 -> 输出结果。"""
    print("=" * 72)
    print("MNIST PCA 降维对比实验")
    print("=" * 72)

    # 1. 加载数据
    print(f"\n[1/3] 加载数据（训练 {_N_TRAIN}，测试 {_N_TEST}）...")
    X_all, y_all, X_t_all, y_t_all = load_mnist("./mnist_data")

    rng = np.random.default_rng(42)
    idx_train = rng.choice(len(X_all), _N_TRAIN, replace=False)
    idx_test = rng.choice(len(X_t_all), _N_TEST, replace=False)

    X_train = X_all[idx_train]
    X_test = X_t_all[idx_test]
    y_train = y_all[idx_train]
    y_test = y_t_all[idx_test]
    print(f"  训练集: {X_train.shape}  测试集: {X_test.shape}")

    # 2. PCA + LR 实验
    print(f"\n[2/3] PCA + LogisticRegression 实验")
    print("-" * 72)
    results: List[Tuple[int, float, float, float]] = []
    for n in _N_COMPONENTS:
        acc, t_pca, t_train = _run_pca_lr(X_train, y_train, X_test, y_test, n)
        results.append((n, acc, t_pca, t_train))
        ratio = n / 784
        print(
            f"  n={n:>3d} ({ratio:.0%} 维度)  "
            f"准确率={acc:.4f}  "
            f"PCA {t_pca:.1f}s + LR {t_train:.1f}s"
        )

    # 3. 汇总
    print(f"\n[3/3] 结果汇总")
    print("=" * 72)
    print(f"{'方法':>20}  {'准确率':>8}  {'用时':>10}")
    print("-" * 72)
    best_acc = max(r[1] for r in results)
    for n, acc, t_pca, t_train in results:
        total_time = t_pca + t_train
        marker = "  ← 最佳" if acc == best_acc else ""
        print(f"  PCA(n={n:>3d}) + LR{marker:>8}  {acc:>8.4f}  {total_time:>6.1f}s")
    print("-" * 72)
    print()


if __name__ == "__main__":
    main()
