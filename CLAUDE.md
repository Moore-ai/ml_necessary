# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

机器学习项目，目前包含 MNIST 手写数字分类任务（`classify_mnist/`）。使用 PyTorch CNN 进行图像分类。

## 代码规范

- **类型注解**：所有函数必须有严格的类型标记（包括参数和返回值），使用 `typing` 模块。
- **代码结构**：避免单文件堆叠过多功能，按模块拆分，使用类组织代码。
- **语言**：注释、文档字符串、提交信息使用中文。
- **检查工具**：每次编写代码后用 Pylance / pyright 检查，确保无类型报错。

## 运行命令

```bash
# CNN 分类（默认 12000 训练样本，2000 测试样本）
python -m classify_mnist.main
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `data_loader.py` | 下载/解析 MNIST 的 IDX 文件，支持展平向量和 CNN 4D 张量两种输出格式 |
| `classifier.py` | `CNNClassifier` — 封装 PyTorch CNN，包含训练、预测、评估接口 |
| `pca.py` | PCA 降维实现（协方差矩阵特征分解），当前未被主流程使用 |
| `main.py` | 入口脚本: 加载数据 → CNN 训练 → 测试评估 |

## 目录结构

```
E:/machine_learning/
├── classify_mnist/    # MNIST 手写数字分类
│   ├── __init__.py
│   ├── data_loader.py   # 数据加载（IDX 解析、采样、格式转换）
│   ├── classifier.py    # CNN 分类器（PyTorch）
│   ├── pca.py           # PCA 实现（备用）
│   └── main.py          # 入口脚本
├── mnist_data/       # MNIST 数据缓存（自动下载）
├── CLAUDE.md
└── .gitignore
```

## CNN 网络结构

```
Conv2d(1, 32, 3×3) → ReLU → MaxPool(2×2)
Conv2d(32, 64, 3×3) → ReLU → MaxPool(2×2)
FC(1600 → 128) → ReLU → FC(128 → 10)
```
