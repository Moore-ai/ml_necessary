"""classify_ALDER 流程图（CNN vs PCA+LR 双路径）。"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# (x, y, w, h, 标题, 描述, 颜色)
DATA_BOX = (0.25, 0.86, 0.50, 0.09,
            "data_loader",  # 标题精简
            "AIDER 数据集 · 5 类 · 224x224 RGB\n8:2 分层抽样 · 0.6x 二次抽样",
            "#E8F5E9")

CNN_BOXES = [
    (0.03, 0.60, 0.42, 0.13,
     "model.AIDERCNN()",
     "4 个 Conv-BN-ReLU-MaxPool 块\n通道: 3→32→64→128→256\nAdaptiveAvgPool + 2 层 FC",
     "#E3F2FD"),
    (0.03, 0.35, 0.42, 0.13,
     "训练循环 (30 epochs)",
     "Weighted CrossEntropyLoss\nAdam + StepLR (step=10, gamma=0.5)\n保存验证集最佳模型",
     "#FFF3E0"),
    (0.03, 0.10, 0.42, 0.13,
     "评估",
     "准确率 / 混淆矩阵 / Kappa / MCC\nPer-class Precision, Recall, F1",
     "#F3E5F5"),
]

LR_BOXES = [
    (0.55, 0.60, 0.42, 0.13,
     "_extract_features()",
     "遍历 DataLoader\n(B,3,224,224) -> (B,150528)\n转移至 CPU NumPy",
     "#E0F2F1"),
    (0.55, 0.35, 0.42, 0.13,
     "PCA + StandardScaler + LR",
     "PCA (randomized, 500 维, 93% 方差)\nStandardScaler 标准化\nLogisticRegression (lbfgs, L2)",
     "#FFF3E0"),
    (0.55, 0.10, 0.42, 0.13,
     "评估 + 对比",
     "与 CNN 相同指标集\n追加 CNN vs PCA+LR 对比表",
     "#F3E5F5"),
]


def _draw_box(ax, x, y, w, h, title, desc, color, edge):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor=color, edgecolor=edge, linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=9, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, desc, ha="center", va="center",
            fontsize=6.5, color="#444")


def _draw_arrow(ax, x1, y1, x2, y2, color="#666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color))


def draw() -> None:
    _, ax = plt.subplots(figsize=(8.5, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 数据加载框
    _draw_box(ax, *DATA_BOX, edge="#333")

    # 分支框
    for box in CNN_BOXES:
        _draw_box(ax, *box, edge="#1565C0")
    for box in LR_BOXES:
        _draw_box(ax, *box, edge="#00897B")

    # 分支标签
    ax.text(0.24, 0.79, "CNN 分支", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#1565C0")
    ax.text(0.76, 0.79, "PCA+LR 分支", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#00897B")

    # 数据加载 → 左右水平分岔
    _draw_arrow(ax, 0.50, 0.86, 0.24, 0.86)
    _draw_arrow(ax, 0.50, 0.86, 0.76, 0.86)

    # 水平 → 向下进入各分支第一框
    _draw_arrow(ax, 0.24, 0.86, 0.24, 0.73, "#1565C0")
    _draw_arrow(ax, 0.76, 0.86, 0.76, 0.73, "#00897B")

    # 分支内部下级箭头
    for lst, color in [(CNN_BOXES, "#1565C0"), (LR_BOXES, "#00897B")]:
        for i in range(len(lst) - 1):
            _, y1, _, _ = lst[i][:4]
            x2, y2, w2, h2 = lst[i + 1][:4]
            cx = x2 + w2 / 2
            _draw_arrow(ax, cx, y1, cx, y2 + h2, color)

    # ax.text(0.50, 0.98, "classify_ALDER  —  CNN vs PCA+LR",
    #         ha="center", va="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("classify_ALDER/flowchart.png", dpi=150, bbox_inches="tight")
    print("已保存 classify_ALDER/flowchart.png")


if __name__ == "__main__":
    draw()
