# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

机器学习项目，包含两个图像分类子项目：
- `classify_mnist/` — MNIST 手写数字分类，PCA 降维 + Softmax 回归
- `classify_ALDER/` — 灾害场景图像分类，PCA+LR 与 CNN 对比实验

## 代码规范

- **类型注解**：所有函数必须有严格的类型标记（包括参数和返回值），使用 `typing` 模块。
- **代码结构**：避免单文件堆叠过多功能，按模块拆分，使用类组织代码。
- **语言**：注释、文档字符串、提交信息使用中文。
- **检查工具**：每次编写代码后用 mypy 检查，确保无类型报错。

## 运行命令

```bash
# MNIST PCA + LogisticRegression 实验
python -m classify_mnist.main

# AIDER PCA + LogisticRegression
python -m classify_ALDER.lr.main

# AIDER CNN（基准线）
python -m classify_ALDER.cnn.main
```

## 模块说明

### classify_mnist/

| 模块 | 职责 |
|------|------|
| `data_loader.py` | 下载/解析 MNIST 的 IDX 文件 |
| `classifier.py` | `LogisticRegression` — Softmax 多项逻辑回归 |
| `pca.py` | PCA 降维实现（协方差矩阵特征分解） |
| `main.py` | 入口脚本: 加载数据 → PCA 降维 → LR 分类 → 输出对比 |

### classify_ALDER/

| 子目录 | 模块 | 职责 |
|--------|------|------|
| `lr/` | `main.py` | PCA(500) + StandardScaler + LogisticRegression |
| `cnn/` | `main.py` | CNN 训练入口 |
| `cnn/` | `model.py` | 卷积神经网络模型定义 |
| `cnn/` | `data_loader.py` | 图像数据加载与增强 |

## 目录结构

```
E:/machine_learning/
├── classify_mnist/       # MNIST 手写数字分类
│   ├── data_loader.py    # IDX 解析、自动下载
│   ├── classifier.py     # Softmax 回归
│   ├── pca.py            # PCA 降维
│   ├── main.py           # 入口脚本
│   └── flowchart.py      # 流水线流程图
│
├── classify_ALDER/       # 灾害场景图像分类
│   ├── lr/               # PCA + LR 方案
│   │   ├── main.py
│   │   └── images/       # 结果可视化
│   └── cnn/              # CNN 基准方案
│       ├── main.py
│       ├── model.py
│       ├── data_loader.py
│       ├── best_aider_cnn.pth
│       └── images/
│
├── mnist_data/           # MNIST 数据缓存（自动下载）
├── docs/
│   └── training-logs/    # 训练日志
├── CLAUDE.md
└── README.md
```
