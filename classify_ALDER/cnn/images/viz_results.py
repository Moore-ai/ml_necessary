"""AIDER CNN 训练结果可视化。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

CLASSES = ["collapsed_building", "fire", "flooded_areas", "normal", "traffic_incident"]
CLASSES_SHORT = ["collapsed", "fire", "flood", "normal", "traffic"]

# per-class 指标
PRECISION = [0.4550, 0.5390, 0.5450, 0.9660, 0.6263]
RECALL = [0.8835, 0.7905, 0.9717, 0.7118, 0.6392]
F1 = [0.6007, 0.6409, 0.6983, 0.8197, 0.6327]
SUPPORT = [103, 105, 106, 878, 97]

# 混淆矩阵
CM = np.array([
    [91, 3, 4, 3, 2],
    [6, 83, 4, 12, 0],
    [0, 0, 103, 1, 2],
    [88, 65, 67, 625, 33],
    [15, 3, 11, 6, 62],
])

# 总体指标
ACC, KAPPA, MCC = 0.7479, 0.5921, 0.6201

# 各类别准确率
CLASS_ACC = [CM[i, i] / CM[i].sum() for i in range(5)]
CLASS_CORRECT = [91, 83, 103, 625, 62]

# 训练曲线（30 epochs）
EPOCHS = list(range(1, 31))
LOSSES = [1.3849, 1.2129, 1.1475, 1.0680, 1.0122, 0.9866, 0.9160, 0.9220,
          0.8830, 0.8226, 0.8171, 0.7930, 0.7827, 0.7346, 0.7367, 0.7120,
          0.7199, 0.6925, 0.7161, 0.7323, 0.6486, 0.6432, 0.6342, 0.6405,
          0.6136, 0.6318, 0.6164, 0.6037, 0.6009, 0.6076]
VAL_ACC = [0.5283, 0.4562, 0.6362, 0.6796, 0.6284, 0.6369, 0.6649, 0.6602,
           0.5997, 0.6043, 0.6858, 0.6734, 0.7393, 0.6850, 0.6897, 0.6920,
           0.6773, 0.6811, 0.6734, 0.6400, 0.7246, 0.6982, 0.6649, 0.7192,
           0.7292, 0.7176, 0.7370, 0.7285, 0.7479, 0.6959]
BEST_EPOCHS = [1, 3, 4, 11, 13, 29]  # 保存最佳模型的 epoch


def plot_training_curve() -> None:
    """图 1: 训练损失 + 验证准确率曲线"""
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color_loss = "#EF5350"
    ax1.plot(EPOCHS, LOSSES, color=color_loss, marker=".", linewidth=1.5,
             markersize=4, label="Train Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10, color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(0.4, 1.6)

    ax2 = ax1.twinx()
    color_acc = "#1565C0"
    ax2.plot(EPOCHS, VAL_ACC, color=color_acc, marker=".", linewidth=1.5,
             markersize=4, label="Val Accuracy")
    ax2.set_ylabel("Accuracy", fontsize=10, color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0.35, 0.85)

    # 最佳模型标记
    for ep in BEST_EPOCHS:
        ax2.scatter(ep, VAL_ACC[ep - 1], color="#FFA000", s=50, zorder=5,
                    edgecolors="#333", linewidths=0.5)

    # 标注最佳准确率
    best_ep = BEST_EPOCHS[-1]
    best_val = VAL_ACC[best_ep - 1]
    ax2.annotate(f"Best: {best_val:.4f}",
                 xy=(best_ep, best_val), xytext=(best_ep + 3, best_val + 0.02),
                 fontsize=9, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#999"))

    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.92), fontsize=9)
    ax1.set_title("Training Curve  (CNN, 30 epochs)", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("classify_ALDER/cnn/training_curve.png", dpi=150)
    print("已保存 training_curve.png")


def plot_per_class() -> None:
    """图 2: 每类的 Precision / Recall / F1 分组柱状图"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CLASSES))
    w = 0.22

    bars_p = ax.bar(x - w, PRECISION, w, label="Precision", color="#42A5F5")
    bars_r = ax.bar(x, RECALL, w, label="Recall", color="#EF5350")
    bars_f = ax.bar(x + w, F1, w, label="F1-score", color="#66BB6A")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class Precision / Recall / F1  (CNN)", fontsize=12, fontweight="bold")

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    for i, s in enumerate(SUPPORT):
        ax.text(i, -0.07, f"n={s}", ha="center", va="top", fontsize=7.5, color="#555")

    plt.tight_layout()
    plt.savefig("classify_ALDER/cnn/per_class_metrics.png", dpi=150)
    print("已保存 per_class_metrics.png")


def plot_confusion_matrix() -> None:
    """图 3: 混淆矩阵热力图"""
    row_sums = CM.sum(axis=1, keepdims=True)
    cm_norm = CM / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    norm = Normalize(vmin=0, vmax=1)
    im = ax.imshow(cm_norm, cmap="Blues", norm=norm)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_yticklabels(CLASSES_SHORT, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix  (row-normalized, CNN)", fontsize=12, fontweight="bold")

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
    plt.savefig("classify_ALDER/cnn/confusion_matrix.png", dpi=150)
    print("已保存 confusion_matrix.png")


def plot_class_accuracy() -> None:
    """图 4: 各类别准确率柱状图"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#EF9A9A", "#EF5350", "#E57373", "#42A5F5", "#FF8A65"]
    bars = ax.bar(CLASSES_SHORT, CLASS_ACC, color=colors, width=0.5,
                  edgecolor="#333", linewidth=0.6)

    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class Accuracy  (CNN)", fontsize=12, fontweight="bold")

    for bar, acc, total, corr in zip(bars, CLASS_ACC, SUPPORT, CLASS_CORRECT):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{acc:.2%}\n({corr}/{total})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("classify_ALDER/cnn/class_accuracy.png", dpi=150)
    print("已保存 class_accuracy.png")


def main() -> None:
    plot_training_curve()
    plot_per_class()
    plot_confusion_matrix()
    plot_class_accuracy()


if __name__ == "__main__":
    main()
