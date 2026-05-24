"""AIDER PCA+LR 训练结果可视化。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
CLASSES_SHORT = ["collapsed", "fire", "flood", "normal", "traffic"]

# per-class 指标
PRECISION = [0.2014, 0.3051, 0.1560, 0.8152, 0.1624]
RECALL = [0.2816, 0.3429, 0.1604, 0.7437, 0.1959]
F1 = [0.2348, 0.3229, 0.1581, 0.7778, 0.1776]
SUPPORT = [103, 105, 106, 878, 97]

# 混淆矩阵
CM = np.array([
    [29, 11, 18, 28, 17],
    [8, 36, 16, 35, 10],
    [13, 11, 17, 50, 15],
    [66, 54, 49, 653, 56],
    [28, 6, 9, 35, 19],
])

# 总体指标
ACC_LR, KAPPA_LR, MCC_LR = 0.5849, 0.2406, 0.2422
ACC_CNN, KAPPA_CNN, MCC_CNN = 0.7479, 0.5921, 0.6201

# 各类准确率
CLASS_ACC = [CM[i, i] / CM[i].sum() for i in range(5)]


def plot_per_class() -> None:
    """图 1: 每类的 Precision / Recall / F1 分组柱状图"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CLASSES))
    w = 0.22

    bars_p = ax.bar(x - w, PRECISION, w, label="Precision", color="#42A5F5")
    bars_r = ax.bar(x, RECALL, w, label="Recall", color="#EF5350")
    bars_f = ax.bar(x + w, F1, w, label="F1-score", color="#66BB6A")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class Precision / Recall / F1  (PCA+LR)", fontsize=12, fontweight="bold")

    # 柱上标注数值
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    # 添加 support 标注
    for i, s in enumerate(SUPPORT):
        ax.text(i, -0.06, f"n={s}", ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    plt.savefig("classify_ALDER/lr/per_class_metrics.png", dpi=150)
    print("已保存 per_class_metrics.png")


def plot_confusion_matrix() -> None:
    """图 2: 混淆矩阵热力图"""
    row_sums = CM.sum(axis=1, keepdims=True)
    cm_norm = CM / row_sums  # 行归一化

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = Normalize(vmin=0, vmax=1)
    im = ax.imshow(cm_norm, cmap="Blues", norm=norm)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_yticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix  (row-normalized)", fontsize=12, fontweight="bold")

    # 格子内标注：原始计数 + 百分比
    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i - 0.10, f"{CM[i, j]}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)
            ax.text(j, i + 0.12, f"({val:.1%})", ha="center", va="center",
                    fontsize=7, color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("classify_ALDER/lr/confusion_matrix.png", dpi=150)
    print("已保存 confusion_matrix.png")


def plot_class_accuracy() -> None:
    """图 3: 各类别准确率柱状图"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#EF9A9A", "#EF5350", "#E57373", "#42A5F5", "#FF8A65"]
    bars = ax.bar(CLASSES_SHORT, CLASS_ACC, color=colors, width=0.5, edgecolor="#333", linewidth=0.6)

    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class Accuracy  (PCA+LR)", fontsize=12, fontweight="bold")

    for bar, acc, total, corr in zip(bars, CLASS_ACC, SUPPORT, CM.diagonal()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{acc:.2%}\n({corr}/{total})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("classify_ALDER/lr/class_accuracy.png", dpi=150)
    print("已保存 class_accuracy.png")


def plot_cnn_comparison() -> None:
    """图 4: CNN vs LR 三指标对比"""
    metrics = ["Accuracy", "Cohen's Kappa", "MCC"]
    cnn_vals = [ACC_CNN, KAPPA_CNN, MCC_CNN]
    lr_vals = [ACC_LR, KAPPA_LR, MCC_LR]

    x = np.arange(len(metrics))
    w = 0.28

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_cnn = ax.bar(x - w / 2, cnn_vals, w, label="CNN", color="#1565C0")
    bars_lr = ax.bar(x + w / 2, lr_vals, w, label="PCA+LR", color="#00897B")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("CNN vs PCA+LR  Overall Comparison", fontsize=12, fontweight="bold")

    for bars, vals in [(bars_cnn, cnn_vals), (bars_lr, lr_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("classify_ALDER/lr/cnn_comparison.png", dpi=150)
    print("已保存 cnn_comparison.png")


def main() -> None:
    plot_per_class()
    plot_confusion_matrix()
    plot_class_accuracy()
    plot_cnn_comparison()


if __name__ == "__main__":
    main()
