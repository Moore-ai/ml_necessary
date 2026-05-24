# 机器学习应用期末报告必选题

基于 PCA 降维 + LogisticRegression（Softmax 回归）的图像分类实验，包含 MNIST 手写数字分类和灾害场景图像分类两个子项目。

[选做题仓库](https://github.com/Moore-ai/ml_optional.git)

---

## 子项目

### 1. `classify_mnist` — MNIST 手写数字分类

从零实现 PCA 降维与 Softmax 回归，对比不同主成分数量对分类准确率的影响。

| 模块 | 职责 |
|------|------|
| `data_loader.py` | 下载/解析 MNIST IDX 文件，归一化至 [0, 1] |
| `pca.py` | PCA 降维（协方差矩阵特征分解），支持正向/反向变换 |
| `classifier.py` | Softmax 回归（L2 正则化 + mini-batch SGD） |
| `main.py` | 入口脚本：加载数据 → PCA → LR → 输出多组维度对比 |
| `flowchart.py` | 生成流水线流程图 |

**运行：**

```bash
python -m classify_mnist.main
```

默认用 12000 训练 / 2000 测试样本，对比 5 ~ 392 维 PCA 的效果。

---

### 2. `classify_ALDER` — 灾害场景图像分类

面向灾害场景的 5 类图像分类任务（collapsed_building, fire, flooded_areas, normal, traffic_incident），基于[AIDER 基准数据集](https://github.com/zhuhong/AIDER)。

包含两种方法对比：

#### 2a. `classify_ALDER/lr` — PCA + LogisticRegression

- PCA 降维（500 分量，保留 ~93% 方差）+ StandardScaler + Softmax 回归
- 训练用时约 40 秒，测试准确率 **58.49%**
- 速度约 CNN 的 **20 倍**，但准确率低约 16 个百分点

#### 2b. `classify_ALDER/cnn` — CNN（基准线）

- 卷积神经网络作为性能基准
- 准确率 **74.79%**，MCC **0.6201**
- 训练用时约 771 秒

**运行：**

```bash
# PCA + LR
python -m classify_ALDER.lr.main

# CNN
python -m classify_ALDER.cnn.main
```

---

## 实验结果小结

| 实验 | 时间 | 准确率 | 备注 |
|------|:----:|:------:|------|
| MNIST PCA(5) + LR | ~5s | 约 0.75 | 仅 5 维已能得到不错结果 |
| MNIST PCA(392) + LR | ~60s | ~0.92 | 保留 50% 维度，接近原特征性能 |
| AIDER PCA(500) + LR | 39.5s | 0.5849 | 速度优势明显，但精度远不如 CNN |
| AIDER CNN | 771.5s | 0.7479 | 代价是训练时间 20 倍 |

> 详细训练日志见 [`docs/training-logs/`](docs/training-logs/)。

---

## 项目结构

```
E:/machine_learning/
├── classify_mnist/           # MNIST 手写数字分类
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载（IDX 解析、采样）
│   ├── classifier.py         # LogisticRegression（Softmax 回归）
│   ├── pca.py                # PCA 降维（协方差特征分解）
│   ├── main.py               # 入口脚本
│   └── flowchart.py          # 流水线流程图
│
├── classify_ALDER/           # 灾害场景图像分类
│   ├── __init__.py
│   ├── flowchart.py
│   ├── lr/                   # PCA + LR 方案
│   │   ├── main.py
│   │   └── images/           # 结果可视化图
│   └── cnn/                  # CNN 基准方案
│       ├── main.py
│       ├── model.py
│       ├── data_loader.py
│       ├── best_aider_cnn.pth  # 预训练权重
│       └── images/           # 结果可视化图
│
├── mnist_data/               # MNIST 数据缓存（自动下载）
├── docs/                     # 文档
│   └── training-logs/        # 训练日志
├── CLAUDE.md                 # Claude Code 项目指令
└── README.md
```

---

## 环境要求

- Python 3.10+
- NumPy
- Matplotlib（用于 `flowchart.py` 和可视化）
- PyTorch（仅 `classify_ALDER/cnn`）

```bash
pip install numpy matplotlib torch
```

---

## 代码规范

- 所有函数必须有严格的类型注解
- 注释、文档字符串、提交信息使用中文
- 提交前使用 mypy 检查类型正确性