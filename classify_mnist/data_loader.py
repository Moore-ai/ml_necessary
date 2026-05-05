"""MNIST 数据集下载与解析模块。

数据来源: Yann LeCun MNIST 数据集
文件格式: IDX (gzip 压缩)
"""

import gzip
import os
import socket
import struct
import urllib.error
import urllib.request
from typing import List, Tuple

import numpy as np

# MNIST 下载源（按优先级排列）
_MNIST_MIRRORS: List[str] = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
]

_MNIST_FILES: dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# IDX 数据类型码 → numpy dtype 映射
_IDX_DTYPE_MAP: dict[int, np.dtype] = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(np.int16),
    0x0C: np.dtype(np.int32),
    0x0D: np.dtype(np.float32),
    0x0E: np.dtype(np.float64),
}


_DOWNLOAD_TIMEOUT: int = 60


def _download_file(url: str, dest_path: str) -> None:
    """下载文件到指定路径。

    Args:
        url: 文件下载地址。
        dest_path: 本地保存路径。

    Raises:
        urllib.error.URLError: 下载失败时抛出。
    """
    print(f"  Downloading {url} ...")
    socket.setdefaulttimeout(_DOWNLOAD_TIMEOUT)
    urllib.request.urlretrieve(url, dest_path)
    # 恢复默认超时
    socket.setdefaulttimeout(None)
    print(f"  Saved to {dest_path}")


def _try_download(
    filename: str, cache_dir: str, mirrors: List[str]
) -> str:
    """依次尝试多个镜像下载文件。

    Args:
        filename: 文件名。
        cache_dir: 缓存目录。
        mirrors: 镜像基址列表（按优先级）。

    Returns:
        下载后文件的完整路径。

    Raises:
        RuntimeError: 所有镜像均失败时抛出。
    """
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        return filepath

    last_error = ""
    for base_url in mirrors:
        url = f"{base_url}/{filename}"
        try:
            _download_file(url, filepath)
            return filepath
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_error = str(e)
            # 清理可能残留的临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
            continue

    raise RuntimeError(
        f"Failed to download {filename} from all mirrors. Last error: {last_error}"
    )


def _parse_idx(filepath: str) -> np.ndarray:
    """解析 IDX 格式文件为 numpy 数组。

    支持 gzip 压缩文件（扩展名为 .gz）。

    Args:
        filepath: IDX 文件路径。

    Returns:
        解析后的 numpy 数组。

    Raises:
        ValueError: 文件格式无效时抛出。
    """
    open_func = gzip.open if filepath.endswith(".gz") else open
    with open_func(filepath, "rb") as f:  # type: ignore[union-attr]
        header = f.read(4)
        if len(header) < 4:
            raise ValueError("Invalid IDX file: header too short")

        zero, dtype_code, ndim = struct.unpack(">HBB", header)
        if zero != 0:
            raise ValueError(f"Invalid IDX magic number: expected 0, got {zero}")

        dims = struct.unpack(f">{'I' * ndim}", f.read(4 * ndim))

        dtype = _IDX_DTYPE_MAP.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unknown IDX data type code: {dtype_code}")

        raw_data = f.read()
        data = np.frombuffer(raw_data, dtype=dtype).reshape(dims)
    return data


def load_mnist(
    cache_dir: str = "./mnist_data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 MNIST 数据集。

    如果本地缓存中没有数据文件，则自动从 Yann LeCun 网站下载。

    Args:
        cache_dir: 数据文件缓存目录。

    Returns:
        (X_train, y_train, X_test, y_test)
        训练集 60000 张图片，测试集 10000 张图片。
        每张图片为 784 维向量，像素值范围 [0, 1]。
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 下载缺失的文件（多镜像容错）
    mirrors = list(_MNIST_MIRRORS)
    paths: dict[str, str] = {}
    for key, filename in _MNIST_FILES.items():
        paths[key] = _try_download(filename, cache_dir, mirrors)

    # 解析文件
    train_images_path = paths["train_images"]
    train_labels_path = paths["train_labels"]
    test_images_path = paths["test_images"]
    test_labels_path = paths["test_labels"]

    X_train_raw = _parse_idx(train_images_path)
    y_train = _parse_idx(train_labels_path)
    X_test_raw = _parse_idx(test_images_path)
    y_test = _parse_idx(test_labels_path)

    # 展平为二维 (n_samples, n_features) 并归一化到 [0, 1]
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).astype(np.float64) / 255.0
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).astype(np.float64) / 255.0

    return X_train, y_train, X_test, y_test
