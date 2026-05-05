# AIDER PCA + LogisticRegression 训练日志

- **日期**: 2026-05-05
- **模型**: PCA(500) + StandardScaler + LogisticRegression (lbfgs, L2)
- **设备**: CPU（PCA/LR 无需 GPU）

---

## 训练配置

| 项目 | 值 |
|------|-----|
| PCA 分量数 | 500（randomized SVD）|
| 保留方差比 | 0.9302 |
| 标准化 | StandardScaler |
| 分类器 | LogisticRegression (lbfgs, C=1.0, L2, class_weight=balanced) |
| 最大迭代 | 1000 |
| 实际迭代 | 58 |
| 训练集抽样 | 0.6×（与 CNN 一致） |

## 数据集

| 类别 | 训练集 | 测试集 |
|------|--------|--------|
| collapsed_building | — | 103 |
| fire | — | 105 |
| flooded_areas | — | 106 |
| normal | — | 878 |
| traffic_incident | — | 97 |
| **合计** | **3086** | **1289** |

> 数据划分与 CNN 实验完全一致（复用 `build_dataloaders`，8:2 分层抽样，seed=42）。

## 流水线用时

| 阶段 | 用时 |
|------|:----:|
| 数据加载 | 0.0s |
| 图像展平 (3086×150528) | 16.8s |
| PCA 降维 (500 分量) | 22.3s |
| 标准化 + LR 训练 | 0.4s |
| **总计** | **39.5s** |

## 最终评估结果

### Per-class 指标

| 类别 | Precision | Recall | F1-score | Support |
|------|:---------:|:------:|:--------:|:-------:|
| collapsed_building | 0.2014 | 0.2816 | 0.2348 | 103 |
| fire | 0.3051 | 0.3429 | 0.3229 | 105 |
| flooded_areas | 0.1560 | 0.1604 | 0.1581 | 106 |
| normal | 0.8152 | 0.7437 | 0.7778 | 878 |
| traffic_incident | 0.1624 | 0.1959 | 0.1776 | 97 |
| **accuracy** | | | **0.5849** | 1289 |
| macro avg | 0.3280 | 0.3449 | 0.3342 | 1289 |
| weighted avg | 0.6213 | 0.5849 | 0.6013 | 1289 |

### 混淆矩阵

| 真实 \ 预测 | collapsed | fire | flood | normal | traffic |
|:------------:|:---------:|:----:|:-----:|:------:|:-------:|
| collapsed_building | 29 | 11 | 18 | 28 | 17 |
| fire | 8 | 36 | 16 | 35 | 10 |
| flooded_areas | 13 | 11 | 17 | 50 | 15 |
| normal | 66 | 54 | 49 | 653 | 56 |
| traffic_incident | 28 | 6 | 9 | 35 | 19 |

### 各类别准确率

| 类别 | 准确率 | 正确/总数 |
|------|:------:|:---------:|
| collapsed_building | 0.2816 | 29/103 |
| fire | 0.3429 | 36/105 |
| flooded_areas | 0.1604 | 17/106 |
| normal | 0.7437 | 653/878 |
| traffic_incident | 0.1959 | 19/97 |

### 总体指标

| 指标 | 值 |
|------|:---:|
| **Overall Accuracy** | **0.5849** |
| **Cohen's Kappa** | **0.2406** |
| **MCC** | **0.2422** |

---

## CNN vs PCA+LR 对比

| 指标 | CNN | PCA+LR | 差值 |
|------|:---:|:------:|:----:|
| 准确率 | 0.7479 | 0.5849 | -0.1630 |
| Cohen's Kappa | 0.5921 | 0.2406 | -0.3515 |
| MCC | 0.6201 | 0.2422 | -0.3779 |
| 训练用时 | 771.5s | 39.5s | ~20× 更快 |

## 分析

1. **PCA+LR 远不如 CNN**：准确率低 16.3 个百分点，Kappa 和 MCC 更是腰斩。CNN 学习层次化空间特征的能力远强于全局像素协方差。

2. **类别不平衡影响更严重**：相比 CNN，LR 对 minority 类别的 recall 更低（如 flooded_areas recall=0.1604 对比 CNN 的 0.9717）。线性模型无法从少数类样本中学习有效决策边界。

3. **normal 类仍占主导**：normal 的 recall 0.7437 尚可，但 precision 0.8152 意味着很多灾害图像被误判为 normal——与 CNN 的安全方向偏差相反，LR 在这方面表现更差。

4. **速度优势明显**：39.5 秒 vs 771 秒，约 20 倍加速。但这是以牺牲大量准确率为代价的。

5. **flooded_areas 表现最差**：recall 仅 0.1604，F1 仅 0.1581。50/106 的 flooded_areas 被误判为 normal，说明水体区域的像素协方差模式与 normal 场景（如绿地、建筑区）混淆严重。
