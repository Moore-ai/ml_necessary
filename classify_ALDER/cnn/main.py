"""AIDER CNN 分类主脚本。

加载数据 → 训练 → 评估 → 保存模型。
"""

import time
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report  # type: ignore[import-untyped]

from classify_ALDER.cnn.data_loader import build_dataloaders, AIDER_CLASSES
from classify_ALDER.cnn.model import AIDERCNN

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "AIDER_data", "AIDER", "AIDER"
)
_BATCH_SIZE = 64
_EPOCHS = 30
_LEARNING_RATE = 1e-3
_WEIGHT_DECAY = 1e-4
_LR_DECAY_STEP = 10
_LR_DECAY_GAMMA = 0.5
_MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "best_aider_cnn.pth"
)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch。

    Args:
        model: 模型。
        loader: 训练 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 设备。

    Returns:
        平均训练损失。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[Tuple[torch.Tensor, int]],
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """评估模型。

    Args:
        model: 模型。
        loader: DataLoader。
        device: 设备。

    Returns:
        (准确率, 所有预测标签, 所有真实标签)。
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, np.array(all_preds), np.array(all_labels)


def main() -> None:
    """主流程: 加载数据 → 训练 → 评估 → 保存模型。"""
    print("=" * 60)
    print("AIDER CNN 分类实验")
    print("=" * 60)
    print(f"设备: {_DEVICE}")

    # 1. 加载数据
    print(f"\n[1/3] 加载数据（8:2 分层抽样）...")
    t0 = time.perf_counter()
    train_loader, test_loader, class_weight = build_dataloaders(
        data_dir=_DATA_DIR, batch_size=_BATCH_SIZE,
    )
    t_load = time.perf_counter() - t0
    print(f"  训练集: {len(train_loader.dataset)} 张")  # type: ignore[arg-type]
    print(f"  测试集: {len(test_loader.dataset)} 张")  # type: ignore[arg-type]
    print(f"  类别: {', '.join(AIDER_CLASSES)}")
    print(f"  加载用时: {t_load:.1f}s")

    # 2. 训练
    print(f"\n[2/3] 训练 CNN（{_EPOCHS} epochs）...")
    model = AIDERCNN(n_classes=len(AIDER_CLASSES)).to(_DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(_DEVICE))
    optimizer = optim.Adam(
        model.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=_LR_DECAY_STEP, gamma=_LR_DECAY_GAMMA
    )

    best_acc = 0.0
    t_train_start = time.perf_counter()

    for epoch in range(1, _EPOCHS + 1):
        t_epoch = time.perf_counter()
        loss = _train_one_epoch(model, train_loader, criterion, optimizer, _DEVICE)
        val_acc, _, _ = _evaluate(model, test_loader, _DEVICE)
        scheduler.step()
        t_epoch = time.perf_counter() - t_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), _MODEL_SAVE_PATH)

        marker = "  ← 最佳模型已保存" if is_best else ""
        print(
            f"  Epoch [{epoch:>2}/{_EPOCHS}]  "
            f"损失={loss:.4f}  准确率={val_acc:.4f}{marker}"
        )

    t_train = time.perf_counter() - t_train_start

    # 3. 加载最佳模型评估
    print(f"\n[3/3] 评估最佳模型...")
    model.load_state_dict(
        torch.load(_MODEL_SAVE_PATH, weights_only=True)
    )
    test_acc, all_preds, all_labels = _evaluate(model, test_loader, _DEVICE)

    # 4. 输出结果
    print("=" * 60)
    print("测试集结果")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=AIDER_CLASSES,
        digits=4,
    ))
    print(f"总体准确率: {test_acc:.4f}")
    print(f"总训练用时: {t_train:.1f}s")
    print(f"模型已保存到: {_MODEL_SAVE_PATH}")
    print()


if __name__ == "__main__":
    main()
