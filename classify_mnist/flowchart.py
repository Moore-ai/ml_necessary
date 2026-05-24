"""classify_mnist 流程图。"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# 模块配置: (x, y, w, h, 标题, 说明, 颜色)
BOXES = [
    (0.15, 0.80, 0.70, 0.10, "data_loader.load_mnist()",
     "下载并解析 IDX -> 784 维", "#E8F5E9"),
    (0.15, 0.60, 0.70, 0.10, "PCA.fit_transform()",
     "协方差特征分解 -> 降维", "#E3F2FD"),
    (0.15, 0.40, 0.70, 0.10, "LogisticRegression.fit()",
     "Softmax 回归 SGD 训练", "#FFF3E0"),
    (0.15, 0.20, 0.70, 0.10, "LogisticRegression.score()",
     "测试集准确率 -> 对比", "#F3E5F5"),
]

ARROWS = [(0, 1), (1, 2), (2, 3)]


def draw() -> None:
    _, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for x, y, w, h, title, desc, color in BOXES:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, desc, ha="center", va="center",
                fontsize=8, color="#555")

    for src, dst in ARROWS:
        _, y_s, _, _ = BOXES[src][:4]
        x_d, y_d, w_d, h_d = BOXES[dst][:4]
        ax.annotate("", xy=(x_d + w_d / 2, y_d + h_d),
                    xytext=(0.50, y_s),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#666"))

    # ax.text(0.50, 0.96, "classify_mnist 流程图",
    #         ha="center", va="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("classify_mnist/flowchart.png", dpi=150, bbox_inches="tight")
    print("已保存 classify_mnist/flowchart.png")


if __name__ == "__main__":
    draw()
