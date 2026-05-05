"""MNIST 手写数字 CNN 分类主脚本。

流程:
  1. 加载 MNIST 数据集（使用子集降低计算量）
  2. 使用 CNN 在原始 2D 图像上训练并分类
  3. 输出测试准确率
"""

import time

from classify_mnist.classifier import CNNClassifier
from classify_mnist.data_loader import load_mnist_cnn


def main() -> None:
    """主流程: 加载数据 -> CNN 训练 -> 预测评估。"""
    print("=" * 60)
    print("MNIST CNN 手写数字分类")
    print("=" * 60)

    # 1. 加载数据（减少样本量以加速）
    n_train = 12000
    n_test = 2000
    print(f"\n[1/3] 加载 MNIST 数据集（训练 {n_train}，测试 {n_test}）...")
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test = load_mnist_cnn(
        cache_dir="./mnist_data", n_train=n_train, n_test=n_test
    )
    t_load = time.perf_counter() - t0
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  数据加载用时 {t_load:.1f}s")

    # 2. CNN 训练
    print(f"\n[2/3] CNN 训练 ...")
    print("-" * 60)
    t0 = time.perf_counter()
    clf = CNNClassifier(batch_size=64, epochs=10, lr=0.001)
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    print(f"  训练总用时 {t_train:.1f}s")

    # 3. 测试评估
    print(f"\n[3/3] 测试评估 ...")
    t0 = time.perf_counter()
    acc = clf.score(X_test, y_test)
    t_test = time.perf_counter() - t0
    print("-" * 60)
    print(f"  测试准确率: {acc:.4f}  (用时 {t_test*1000:.0f}ms)")
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
