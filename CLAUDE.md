# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

机器学习项目，目前包含 MNIST 手写数字降维与分类任务（`classify_mnist/`）。使用 PCA 降维 + LogisticRegression（Softmax 回归）进行分类。

## 代码规范

- **类型注解**：所有函数必须有严格的类型标记（包括参数和返回值），使用 `typing` 模块。
- **代码结构**：避免单文件堆叠过多功能，按模块拆分，使用类组织代码。
- **语言**：注释、文档字符串、提交信息使用中文。
- **检查工具**：每次编写代码后用 mypy 检查，确保无类型报错。

## 运行命令

```bash
# PCA + LogisticRegression 降维分类实验
python -m classify_mnist.main
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `data_loader.py` | 下载/解析 MNIST 的 IDX 文件 |
| `classifier.py` | `LogisticRegression` — Softmax 多项逻辑回归 |
| `pca.py` | PCA 降维实现（协方差矩阵特征分解） |
| `main.py` | 入口脚本: 加载数据 → PCA 降维 → LR 分类 → 输出对比 |

## 目录结构

```
E:/machine_learning/
├── classify_mnist/    # MNIST 手写数字分类
│   ├── __init__.py
│   ├── data_loader.py   # 数据加载（IDX 解析、采样）
│   ├── classifier.py    # LogisticRegression 分类器
│   ├── pca.py           # PCA 降维实现
│   └── main.py          # 入口脚本
├── mnist_data/       # MNIST 数据缓存（自动下载）
├── CLAUDE.md
└── .gitignore
```
