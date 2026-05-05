"""AIDER 数据集加载与预处理模块。

提供 AIDERDataset 类和训练/测试 DataLoader 的构建函数。
"""

import os
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # type: ignore[import-untyped]


# AIDER 类别名称（按字母顺序排列）
AIDER_CLASSES: List[str] = [
    "collapsed_building",
    "fire",
    "flooded_areas",
    "normal",
    "traffic_incident",
]

# Normalize 参数（ImageNet 统计量）
_NORMALIZE_MEAN: List[float] = [0.485, 0.456, 0.406]
_NORMALIZE_STD: List[float] = [0.229, 0.224, 0.225]


class AIDERDataset(Dataset[Tuple[torch.Tensor, int]]):
    """AIDER 航空应急图像数据集。

    Attributes:
        image_paths: 所有图像文件路径列表。
        labels: 对应的标签列表。
        class_to_idx: 类别名到标签索引的映射。
        transform: 图像预处理/增强变换。
    """

    def __init__(self, data_dir: str, transform: transforms.Compose | None = None) -> None:
        """扫描 AIDER 数据集目录，建立 (image_path, label) 映射。

        Args:
            data_dir: 数据根目录，包含 5 个类别子文件夹。
            transform: 图像变换流水线。若为 None，则仅做 ToTensor。

        Raises:
            RuntimeError: 目录中未找到任何有效图像文件时抛出。
        """
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: dict[str, int] = {
            cls: i for i, cls in enumerate(AIDER_CLASSES)
        }
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_idx)

        if len(self.labels) == 0:
            raise RuntimeError(
                f"AIDER 数据集目录 '{data_dir}' 下未找到任何有效图像文件"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image)
        return img_tensor, self.labels[idx]


def _train_transform() -> transforms.Compose:
    """训练集数据增强 + 归一化。

    Returns:
        训练图像变换流水线。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
    ])


def _test_transform() -> transforms.Compose:
    """测试集仅缩放 + 归一化。

    Returns:
        测试图像变换流水线。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
    ])


def _stratified_split(
    labels: List[int],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """分层抽样划分训练集和测试集索引。

    Args:
        labels: 全部样本的标签列表。
        train_ratio: 训练集比例。
        random_seed: 随机种子。

    Returns:
        (train_indices, test_indices) 索引列表。
    """
    rng = np.random.default_rng(random_seed)
    label_arr = np.array(labels)
    classes = np.unique(label_arr)
    train_idx: List[int] = []
    test_idx: List[int] = []

    for cls in classes:
        cls_indices = np.where(label_arr == cls)[0]
        rng.shuffle(cls_indices)
        n_train = int(len(cls_indices) * train_ratio)
        train_idx.extend(cls_indices[:n_train].tolist())
        test_idx.extend(cls_indices[n_train:].tolist())

    return train_idx, test_idx


def _compute_class_weight(labels: List[int], n_classes: int) -> torch.Tensor:
    """计算类别权重，用于 WeightedCrossEntropyLoss。

    weight[i] = total_samples / (n_classes * samples_per_class[i])

    Args:
        labels: 全部训练标签。
        n_classes: 类别总数。

    Returns:
        形状 (n_classes,) 的权重张量。
    """
    label_arr = np.array(labels)
    weights = np.zeros(n_classes, dtype=np.float64)
    for i in range(n_classes):
        count = int(np.sum(label_arr == i))
        weights[i] = len(labels) / (n_classes * count) if count > 0 else 1.0
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    num_workers: int = 0,
) -> Tuple[DataLoader[Tuple[torch.Tensor, int]], DataLoader[Tuple[torch.Tensor, int]], torch.Tensor]:
    """构建训练集和测试集 DataLoader。

    Args:
        data_dir: AIDER 数据根目录（包含 5 个类别子文件夹）。
        batch_size: 每批样本数。
        train_ratio: 训练集比例。
        num_workers: DataLoader 工作进程数。

    Returns:
        (train_loader, test_loader, class_weight) 元组。
    """
    # 仅扫描一次目录，通过切片避免重复扫描
    full_dataset = AIDERDataset(data_dir, transform=None)

    train_idx, test_idx = _stratified_split(
        full_dataset.labels, train_ratio=train_ratio
    )

    # 从 full_dataset 切片，不重新扫描目录
    train_dataset = AIDERDataset.__new__(AIDERDataset)
    train_dataset.image_paths = [full_dataset.image_paths[i] for i in train_idx]
    train_dataset.labels = [full_dataset.labels[i] for i in train_idx]
    train_dataset.class_to_idx = full_dataset.class_to_idx
    train_dataset.transform = _train_transform()

    test_dataset = AIDERDataset.__new__(AIDERDataset)
    test_dataset.image_paths = [full_dataset.image_paths[i] for i in test_idx]
    test_dataset.labels = [full_dataset.labels[i] for i in test_idx]
    test_dataset.class_to_idx = full_dataset.class_to_idx
    test_dataset.transform = _test_transform()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_weight = _compute_class_weight(train_dataset.labels, len(AIDER_CLASSES))

    return train_loader, test_loader, class_weight
