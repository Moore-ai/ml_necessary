"""AIDER PCA + LogisticRegression 分类主脚本。

加载数据 → 展平 → PCA 降维 → LogisticRegression → 评估 → 与 CNN 对比。
"""

import time
import os
from typing import Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
)

from classify_ALDER.cnn.data_loader import build_dataloaders, AIDER_CLASSES


_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "AIDER_data", "AIDER", "AIDER"
)
_BATCH_SIZE = 64


def _extract_features(
    loader: DataLoader[Tuple[torch.Tensor, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """遍历 DataLoader，将图像展平为 (N, 150528) 特征矩阵。

    Args:
        loader: DataLoader，返回 (images, labels)。

    Returns:
        (feature_matrix, labels) 元组。
    """
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []

    for images, labels in loader:
        # (B, 3, 224, 224) → (B, 150528); PCA/LR 在 CPU 上运行，无需 GPU 传输
        batch = images.view(images.size(0), -1).numpy()
        all_features.append(batch)
        all_labels.extend(labels.tolist())

    features = np.concatenate(all_features, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)
    return features, labels_arr


def main() -> None:
    """主流程: 加载数据 → 展平 → PCA → LR → 评估 → 对比。"""
    print("=" * 60)
    print("AIDER PCA + LogisticRegression 分类实验")
    print("=" * 60)

    # 1. 加载数据（复用 CNN 的 build_dataloaders，保证完全一致的划分）
    print(f"\n[1/5] 加载数据（8:2 分层抽样，0.6× 训练集二次抽样）...")
    t0 = time.perf_counter()
    train_loader, test_loader, _ = build_dataloaders(
        data_dir=_DATA_DIR, batch_size=_BATCH_SIZE, train_subsample=0.6,
    )
    t_load = time.perf_counter() - t0
    n_train = len(cast(torch.utils.data.dataset.Subset, train_loader.dataset))  # type: ignore[arg-type]
    n_test = len(cast(torch.utils.data.dataset.Subset, test_loader.dataset))  # type: ignore[arg-type]
    print(f"  训练集: {n_train} 张")
    print(f"  测试集: {n_test} 张")
    print(f"  类别: {', '.join(AIDER_CLASSES)}")
    print(f"  加载用时: {t_load:.1f}s")

    # 2. 展平图像
    print(f"\n[2/5] 展平图像 (3×224×224 → 150528) 并转移到 CPU...")
    t0 = time.perf_counter()
    X_train, y_train = _extract_features(train_loader)
    X_test, y_test = _extract_features(test_loader)
    t_feat = time.perf_counter() - t0
    print(f"  训练集特征矩阵: {X_train.shape}")
    print(f"  测试集特征矩阵: {X_test.shape}")
    print(f"  展平用时: {t_feat:.1f}s")

    # 3. PCA 降维（保留 95% 方差）
    print(f"\n[3/5] PCA 降维（保留 95% 方差）...")
    t0 = time.perf_counter()
    pca = PCA(n_components=0.95, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    t_pca = time.perf_counter() - t0
    print(f"  降维后维度: {X_train_pca.shape[1]}")
    print(f"  保留方差比: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"  PCA 用时: {t_pca:.1f}s")

    # 4. LogisticRegression 训练（Softmax 多分类）
    print(f"\n[4/5] 训练 LogisticRegression（Softmax，L2 正则化）...")
    t0 = time.perf_counter()
    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_train_pca, y_train)
    t_train = time.perf_counter() - t0
    print(f"  收敛迭代次数: {lr.n_iter_[0]}")
    print(f"  训练用时: {t_train:.1f}s")

    # 5. 评估
    print(f"\n[5/5] 评估模型...")
    y_pred = lr.predict(X_test_pca)
    test_acc = (y_pred == y_test).mean()

    print("=" * 60)
    print("测试集结果")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=AIDER_CLASSES,
        digits=4,
    ))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵")
    print(" " * 22 + "  " + "  ".join(f"{c:>10}" for c in AIDER_CLASSES))
    for i, cls_name in enumerate(AIDER_CLASSES):
        row = "  ".join(f"{cm[i, j]:>10}" for j in range(len(AIDER_CLASSES)))
        print(f"  {cls_name:>20}  {row}")

    # 各类别准确率
    print("\n各类别准确率:")
    for i, cls_name in enumerate(AIDER_CLASSES):
        cls_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
        print(f"  {cls_name:>20}: {cls_acc:.4f} ({cm[i, i]}/{cm[i, :].sum()})")

    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"\n总体准确率: {test_acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"MCC:           {mcc:.4f}")
    print(f"总用时: {t_load + t_feat + t_pca + t_train:.1f}s")

    # 6. 与 CNN 结果对比
    # CNN 数据来源：0.6× 训练集二次抽样实验，commit 901dd8a，详见 docs/training-logs/2026-05-05-AIDER-CNN-training-log.md
    CNN_RESULTS = {
        "accuracy": 0.7479,
        "kappa": 0.5921,
        "mcc": 0.6201,
    }
    print("\n" + "=" * 60)
    print("CNN vs PCA+LR 对比")
    print("=" * 60)
    print(f"  {'指标':<20} {'CNN':<12} {'PCA+LR':<12} {'差值':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'准确率':<20} {CNN_RESULTS['accuracy']:<12.4f} {test_acc:<12.4f} {test_acc - CNN_RESULTS['accuracy']:<+12.4f}")
    print(f"  {'Kappa':<20} {CNN_RESULTS['kappa']:<12.4f} {kappa:<12.4f} {kappa - CNN_RESULTS['kappa']:<+12.4f}")
    print(f"  {'MCC':<20} {CNN_RESULTS['mcc']:<12.4f} {mcc:<12.4f} {mcc - CNN_RESULTS['mcc']:<+12.4f}")
    print()


if __name__ == "__main__":
    main()
