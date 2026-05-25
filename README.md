# 机器学习应用期末报告必选题

基于 PCA 降维与 LogisticRegression（Softmax 回归）的图像分类实验，包含 MNIST 手写数字分类和 AIDER 灾害场景图像分类两个子项目。

[选做题仓库](https://github.com/Moore-ai/ml_optional.git)

---

## 项目背景

本项目是机器学习应用课程的期末必选作业，主要探索两种经典的图像分类方法：

1. **传统机器学习方法**：PCA 降维 + LogisticRegression（Softmax 回归）
2. **深度学习方法**：卷积神经网络（CNN）

通过两个不同规模和复杂度的数据集（MNIST 和 AIDER），对比分析两种方法的性能、训练效率和适用场景。

---

## 子项目一：MNIST 手写数字分类

### 项目简介

从零实现 PCA 降维与 Softmax 回归，对比不同主成分数量对分类准确率的影响。MNIST 数据集包含 0-9 共 10 个类别的手写数字灰度图像（28×28 像素），训练集 60000 张，测试集 10000 张。

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据加载 | `data_loader.py` | 下载/解析 MNIST IDX 文件，归一化至 [0, 1] |
| PCA 降维 | `pca.py` | 协方差矩阵特征分解实现，支持 fit/transform/inverse_transform |
| Softmax 回归 | `classifier.py` | L2 正则化 + mini-batch SGD 优化 |
| 主脚本 | `main.py` | 多组 PCA 维度对比实验 |
| 流程图 | `flowchart.py` | 生成流水线流程图 |

### 算法实现

#### PCA（主成分分析）

- **方法**：协方差矩阵特征分解（对 n_features < n_samples 场景高效）
- **步骤**：中心化 → 计算协方差矩阵 → 特征分解 → 取前 k 个主成分
- **特点**：支持正向变换（降维）和反向变换（重建）

#### LogisticRegression（Softmax 回归）

- **损失函数**：交叉熵 + L2 正则化
- **优化器**：Mini-batch 梯度下降
- **超参数**：learning_rate=0.1, max_iter=200, batch_size=256

### 运行方法

```bash
python -m classify_mnist.main
```

默认配置：
- 训练样本：12000（随机抽样）
- 测试样本：2000（随机抽样）
- PCA 维度：[5, 10, 20, 50, 100, 200, 392]

### 实验结果

| PCA 维度 | 保留方差比 | 准确率 | PCA 用时 | 训练用时 |
|:--------:|:----------:|:------:|:--------:|:--------:|
| 5 | 16.9% | ~0.75 | <1s | ~2s |
| 10 | 27.3% | ~0.81 | <1s | ~2s |
| 20 | 38.5% | ~0.86 | <1s | ~3s |
| 50 | 56.3% | ~0.89 | ~1s | ~5s |
| 100 | 71.5% | ~0.91 | ~2s | ~8s |
| 200 | 84.2% | ~0.92 | ~4s | ~15s |
| 392 | 100% | ~0.92 | ~8s | ~30s |

**关键发现**：
- 仅 5 维特征即可达到 75% 准确率
- 100 维以上准确率趋于稳定（~91%）
- 392 维（50% 原维度）已接近原特征性能

---

## 子项目二：AIDER 灾害场景图像分类

### 项目简介

面向航空应急场景的 5 类图像分类任务，基于 [AIDER 基准数据集](https://github.com/zhuhong/AIDER)。对比传统机器学习方法（PCA+LR）与深度学习方法（CNN）的性能差异。

**类别**：
- collapsed_building（建筑倒塌）
- fire（火灾）
- flooded_areas（洪水淹没）
- normal（正常场景）
- traffic_incident（交通事故）

### 数据集统计

| 类别 | 训练集 | 测试集 |
|------|:------:|:------:|
| collapsed_building | — | 103 |
| fire | — | 105 |
| flooded_areas | — | 106 |
| normal | — | 878 |
| traffic_incident | — | 97 |
| **合计** | **3086*** | **1289** |

> *训练集使用 0.6× 二次抽样（全量 5144 张），测试集为全量。数据划分采用 8:2 分层抽样。

### 方法对比

#### 方法一：PCA + LogisticRegression

**目录**：`classify_ALDER/lr/`

**流水线**：
1. 图像展平：3×224×224 → 150528 维向量
2. PCA 降维：randomized SVD，500 分量（保留 ~93% 方差）
3. 标准化：StandardScaler
4. 分类器：LogisticRegression（lbfgs, L2 正则化, class_weight=balanced）

**运行**：
```bash
python -m classify_ALDER.lr.main
```

**训练用时**：约 40 秒（CPU）

#### 方法二：卷积神经网络（CNN）

**目录**：`classify_ALDER/cnn/`

**网络架构**：
```
AIDERCNN(
  features: 4 × Conv-BN-ReLU-MaxPool 块
    Block1: 3 → 32, 224 → 112
    Block2: 32 → 64, 112 → 56
    Block3: 64 → 128, 56 → 28
    Block4: 128 → 256, 28 → 14
  
  classifier:
    AdaptiveAvgPool2d(1)
    Dropout(0.5)
    Linear(256 → 128) + ReLU
    Dropout(0.5)
    Linear(128 → 5)
)
```

**训练配置**：
- 损失函数：Weighted CrossEntropyLoss
- 优化器：Adam (lr=1e-3, weight_decay=1e-4)
- 学习率衰减：StepLR (step=10, gamma=0.5)
- Batch size：64
- Epochs：30
- 数据增强：RandomHorizontalFlip(p=0.5) + RandomRotation(±10°)

**运行**：
```bash
python -m classify_ALDER.cnn.main
```

**训练用时**：约 771 秒（NVIDIA RTX 4060 Laptop GPU）

### 详细结果对比

#### 总体指标

| 指标 | CNN | PCA+LR | 差值 |
|------|:---:|:------:|:----:|
| **准确率** | **74.79%** | 58.49% | -16.30% |
| **Cohen's Kappa** | **0.5921** | 0.2406 | -0.3515 |
| **MCC** | **0.6201** | 0.2422 | -0.3779 |
| **训练用时** | 771.5s | 39.5s | 20× 更慢 |

#### 各类别性能对比

| 类别 | CNN Recall | LR Recall | CNN F1 | LR F1 |
|------|:----------:|:---------:|:------:|:-----:|
| collapsed_building | 0.8835 | 0.2816 | 0.6007 | 0.2348 |
| fire | 0.7905 | 0.3429 | 0.6409 | 0.3229 |
| flooded_areas | **0.9717** | 0.1604 | **0.6983** | 0.1581 |
| normal | 0.7118 | 0.7437 | 0.8197 | 0.7778 |
| traffic_incident | 0.6392 | 0.1959 | 0.6327 | 0.1776 |

#### 混淆矩阵（CNN）

| 真实 \ 预测 | collapsed | fire | flood | normal | traffic |
|:-----------:|:---------:|:----:|:-----:|:------:|:-------:|
| collapsed_building | 91 | 3 | 4 | 3 | 2 |
| fire | 6 | 83 | 4 | 12 | 0 |
| flooded_areas | 0 | 0 | 103 | 1 | 2 |
| normal | 88 | 65 | 67 | 625 | 33 |
| traffic_incident | 15 | 3 | 11 | 6 | 62 |

### 关键发现

1. **CNN 显著优于 PCA+LR**：准确率高出 16.3 个百分点，Kappa 和 MCC 更是翻倍。CNN 能学习层次化的空间特征，而 PCA 仅捕获全局像素协方差。

2. **类别不平衡影响**：
   - CNN：对 minority 类（collapsed, fire, flood, traffic）recall 达 0.64-0.97，normal recall 仅 0.71
   - LR：normal recall 0.74，但 minority 类 recall 仅 0.16-0.34

3. **安全方向偏差**：CNN 倾向于将 normal 误判为灾害（偏向保守，宁可误报不错报），LR 则反之。

4. **速度与精度权衡**：LR 快 20 倍，但精度差距大；适用于快速原型或资源受限场景。

5. **flooded_areas 差异最大**：CNN recall 0.97，LR 仅 0.16，说明水体区域需要 CNN 学习的空间特征。

> 详细训练日志见 [`docs/training-logs/`](docs/training-logs/)。

---

## 项目结构

```
E:/machine_learning/
├── classify_mnist/               # MNIST 手写数字分类
│   ├── __init__.py
│   ├── data_loader.py            # IDX 解析、自动下载、归一化
│   ├── classifier.py             # LogisticRegression（Softmax 回归）
│   ├── pca.py                    # PCA 降维（协方差特征分解）
│   ├── main.py                   # 入口脚本
│   ├── flowchart.py              # 流水线流程图生成
│   └── flowchart.png             # 流程图
│
├── classify_ALDER/               # AIDER 灾害场景图像分类
│   ├── __init__.py
│   ├── flowchart.py
│   ├── flowchart.png
│   │
│   ├── lr/                       # 方法一：PCA + LogisticRegression
│   │   ├── __init__.py
│   │   ├── main.py               # LR 入口脚本
│   │   └── images/               # 结果可视化
│   │       ├── confusion_matrix.png
│   │       ├── class_accuracy.png
│   │       ├── per_class_metrics.png
│   │       └── cnn_comparison.png
│   │
│   └── cnn/                      # 方法二：CNN
│       ├── __init__.py
│       ├── main.py               # CNN 训练入口
│       ├── model.py              # AIDERCNN 网络定义
│       ├── data_loader.py        # 数据加载、增强、分层划分
│       ├── best_aider_cnn.pth    # 预训练权重
│       └── images/               # 结果可视化
│           ├── training_curve.png
│           ├── confusion_matrix.png
│           ├── class_accuracy.png
│           └── per_class_metrics.png
│
├── mnist_data/                   # MNIST 数据缓存（自动下载）
├── AIDER_data/                   # AIDER 数据集（需手动下载）
├── docs/
│   └── training-logs/            # 训练日志
│       ├── 2026-05-05-AIDER-CNN-training-log.md
│       └── 2026-05-05-AIDER-PCA-LR-training-log.md
├── CLAUDE.md                     # Claude Code 项目指令
└── README.md
```

---

## 环境要求

- Python 3.10+
- NumPy
- Matplotlib
- PyTorch（仅 AIDER CNN）
- scikit-learn（仅 AIDER LR）
- Pillow
- torchvision

```bash
pip install numpy matplotlib torch scikit-learn pillow torchvision
```

---

## 数据集准备

### MNIST

自动下载，首次运行时会从镜像源下载到 `mnist_data/` 目录。

### AIDER

1. 从 [AIDER GitHub](https://github.com/zhuhong/AIDER) 下载数据集
2. 解压到 `AIDER_data/AIDER/AIDER/` 目录，结构如下：
   ```
   AIDER_data/AIDER/AIDER/
   ├── collapsed_building/
   ├── fire/
   ├── flooded_areas/
   ├── normal/
   └── traffic_incident/
   ```

---

## 代码规范

- 所有函数必须有严格的类型注解（使用 `typing` 模块）
- 注释、文档字符串、提交信息使用中文
- 提交前使用 mypy 检查类型正确性

```bash
mypy classify_mnist/ classify_ALDER/
```

---

## 参考资料

- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)
- [AIDER 数据集](https://github.com/zhuhong/AIDER)
- [PCA 原理](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Softmax 回归](https://en.wikipedia.org/wiki/Softmax_function)
