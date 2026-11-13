"""
Cross-dataset IED classification.

Four protocols:
1. Pretrain-Finetune
2. Source-Only
3. Target-Only
4. Joint-Training

Common representation:
- 20-channel TCP
- 4 seconds
- 500 Hz
- 2000 samples
- Class 0 = True-IED
- Class 1 = Non-IED
- Fixed classification threshold = 0.5
"""

import copy
import csv
import glob
import importlib.util
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)

from model import CausalIEDGraphNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEEDS = tuple(range(42, 52))

METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auroc",
    "auprc",
)


@dataclass
class Config:
    # ==========================================================
    # 数据路径：根据你的电脑修改
    # ==========================================================
    vepi_data_dir: str = r"D:\Epilepsy-2class"
    tuev_data_root: str = r"D:\TUEV\v2.0.1\edf"

    # CrossDataset.py、model.py和该文件放在同一文件夹。
    tuev_helper_script: str = r"TUEV.py"

    cache_dir: str = r"D:\IED_cross_dataset_cache"
    output_dir: str = r"D:\IED_cross_dataset_results"

    # ==========================================================
    # VEPI前19个通道的实际顺序
    # ==========================================================
    vepi_channel_names: Tuple[str, ...] = (
        "Fp1", "Fp2",
        "F3", "F4",
        "C3", "C4",
        "P3", "P4",
        "O1", "O2",
        "F7", "F8",
        "T3", "T4",
        "T5", "T6",
        "Fz", "Cz", "Pz",
    )

    # 文件名第一个下划线前为患者编号。
    vepi_group_regex: str = r"^([^_]+)_"

    vepi_sfreq: float = 500.0

    # ==========================================================
    # 两个数据集统一后的输入
    # ==========================================================
    channels: int = 20
    epoch_seconds: float = 4.0
    target_len: int = 2000
    time_bins: int = 16

    # ==========================================================
    # 患者级数据划分
    # ==========================================================
    source_val_fraction: float = 0.10

    # TUEV官方train中的10%用于梯度训练。
    tuev_train_fraction: float = 0.10

    # 另外10%仅用于早停。
    target_val_fraction: float = 0.10

    # 剩余80%不使用。
    split_seed: int = 42

    # 正式运行10个随机种子。
    seeds: Tuple[int, ...] = RANDOM_SEEDS

    # ==========================================================
    # 四种实验统一使用的模型参数
    # ==========================================================
    node_dim: int = 24
    graph_hidden: int = 48
    rank_dim: int = 12
    dropout: float = 0.10

    batch_size: int = 128
    num_workers: int = 0
    weight_decay: float = 1e-4

    # ==========================================================
    # 各训练阶段参数
    # ==========================================================
    pretrain_epochs: int = 100
    pretrain_patience: int = 15
    pretrain_lr: float = 1e-3

    finetune_epochs: int = 50
    finetune_patience: int = 10
    finetune_lr: float = 1e-4

    target_epochs: int = 100
    target_patience: int = 15
    target_lr: float = 1e-3

    joint_epochs: int = 100
    joint_patience: int = 15
    joint_lr: float = 1e-3

    fixed_threshold: float = 0.5

    # Joint-Training中两个域具有相同的总采样概率。
    balance_joint_domains: bool = True

    # 修改预处理设置后，将对应变量设为True重建一次。
    rebuild_vepi_cache: bool = False

    # TUEV原代码需要使用这个变量名。
    rebuild_cache: bool = False

    save_checkpoints: bool = False


class CachedVEPIDataset(Dataset):
    """通过memmap读取转换后的VEPI，避免一次加载全部数据。"""

    def __init__(
        self,
        data_path: Path,
        label_path: Path,
        group_path: Path,
    ):
        self.data = np.load(
            data_path,
            mmap_mode="c",
            allow_pickle=False,
        )

        self.labels = np.load(
            label_path,
            allow_pickle=False,
        ).astype(np.int64)

        self.groups = np.load(
            group_path,
            allow_pickle=False,
        ).astype(str)

        if self.data.ndim != 3:
            raise ValueError(
                f"VEPI缓存应为[N,C,L]，实际为{self.data.shape}"
            )

        if not (
            len(self.data)
            == len(self.labels)
            == len(self.groups)
        ):
            raise ValueError(
                "VEPI缓存的data、labels和groups数量不一致"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = np.asarray(
            self.data[index],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(data),
            int(self.labels[index]),
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_tuev_helper(path: Path):
    """加载原TUEV程序中的数据读取代码。"""

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"TUEV脚本不存在：{path}"
        )

    module_name = "_tuev_cross_dataset_helper"

    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"无法加载TUEV脚本：{path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module
    specification.loader.exec_module(module)

    required_names = (
        "TCP_MONTAGE_20",
        "TUEVEpochDataset",
    )

    missing_names = [
        name
        for name in required_names
        if not hasattr(module, name)
    ]

    if missing_names:
        raise AttributeError(
            f"{path.name}缺少以下对象：{missing_names}"
        )

    return module


def normalize_electrode_name(name: str):
    """兼容旧版和新版颞区电极命名。"""

    name = name.upper().strip()

    aliases = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    }

    return aliases.get(name, name)


def convert_vepi_to_tcp(
    data: np.ndarray,
    cfg: Config,
    tcp_montage: Sequence[Tuple[str, str]],
):
    """
    将VEPI转换为与TUEV一致的20通道TCP。

    支持：
    - (29, time)
    - (time, 29)
    - (19, time)
    - (time, 19)

    少于2000点的片段在标准化后居中补0。
    """

    selected_channel_number = len(
        cfg.vepi_channel_names
    )

    if data.ndim != 2:
        raise ValueError(
            f"VEPI样本应为二维数组，实际为{data.shape}"
        )

    possible_channel_numbers = {
        29,
        selected_channel_number,
    }

    if data.shape[0] in possible_channel_numbers:
        # [channels, time]
        pass

    elif data.shape[1] in possible_channel_numbers:
        # [time, channels]
        data = data.T

    else:
        raise ValueError(
            f"无法识别VEPI数据的通道维，"
            f"实际形状为{data.shape}"
        )

    if data.shape[0] < selected_channel_number:
        raise ValueError(
            f"VEPI至少需要{selected_channel_number}个通道，"
            f"实际只有{data.shape[0]}个通道"
        )

    # 与原VEPI代码一致：29通道中只使用前19个。
    data = np.asarray(
        data[:selected_channel_number],
        dtype=np.float32,
    )

    if not np.isfinite(data).all():
        raise ValueError(
            "VEPI前19个通道中存在NaN或Inf"
        )

    # 如果存在超过4秒的片段，则居中截取4秒。
    maximum_source_length = int(
        round(
            cfg.vepi_sfreq
            * cfg.epoch_seconds
        )
    )

    if data.shape[1] > maximum_source_length:
        start = (
            data.shape[1]
            - maximum_source_length
        ) // 2

        data = data[
            :,
            start:start + maximum_source_length,
        ]

    channel_names = [
        normalize_electrode_name(name)
        for name in cfg.vepi_channel_names
    ]

    if len(set(channel_names)) != len(channel_names):
        raise ValueError(
            "VEPI通道列表中存在重复电极"
        )

    channel_index = {
        name: index
        for index, name in enumerate(channel_names)
    }

    required_electrodes = {
        normalize_electrode_name(electrode)
        for pair in tcp_montage
        for electrode in pair
    }

    missing_electrodes = sorted(
        required_electrodes - set(channel_index)
    )

    if missing_electrodes:
        raise ValueError(
            "VEPI缺少构建TCP所需的电极："
            + ", ".join(missing_electrodes)
        )

    # 转换为20通道TCP。
    tcp = np.stack(
        [
            data[
                channel_index[
                    normalize_electrode_name(left)
                ]
            ]
            -
            data[
                channel_index[
                    normalize_electrode_name(right)
                ]
            ]
            for left, right in tcp_montage
        ],
        axis=0,
    )

    tcp = np.asarray(
        tcp,
        dtype=np.float32,
    )

    # 改变采样率，但不把短片段拉伸为4秒。
    target_sfreq = (
        cfg.target_len
        / cfg.epoch_seconds
    )

    if not np.isclose(
        target_sfreq,
        cfg.vepi_sfreq,
    ):
        resampled_length = max(
            1,
            int(
                round(
                    tcp.shape[1]
                    * target_sfreq
                    / cfg.vepi_sfreq
                )
            ),
        )

        tcp = resample(
            tcp,
            resampled_length,
            axis=1,
        )

        tcp = np.asarray(
            tcp,
            dtype=np.float32,
        )

    # 先标准化有效信号。
    tcp = (
        tcp
        - tcp.mean(axis=1, keepdims=True)
    ) / np.maximum(
        tcp.std(axis=1, keepdims=True),
        1e-6,
    )

    # 再将短信号居中补0。
    if tcp.shape[1] < cfg.target_len:
        padding = (
            cfg.target_len
            - tcp.shape[1]
        )

        left_padding = padding // 2
        right_padding = (
            padding - left_padding
        )

        tcp = np.pad(
            tcp,
            (
                (0, 0),
                (
                    left_padding,
                    right_padding,
                ),
            ),
            mode="constant",
            constant_values=0.0,
        )

    elif tcp.shape[1] > cfg.target_len:
        start = (
            tcp.shape[1]
            - cfg.target_len
        ) // 2

        tcp = tcp[
            :,
            start:start + cfg.target_len,
        ]

    expected_shape = (
        cfg.channels,
        cfg.target_len,
    )

    if tcp.shape != expected_shape:
        raise RuntimeError(
            f"VEPI转换后的形状为{tcp.shape}，"
            f"预期为{expected_shape}"
        )

    return np.ascontiguousarray(
        tcp,
        dtype=np.float32,
    )


def load_or_build_vepi(
    cfg: Config,
    tcp_montage,
):
    """
    将VEPI转换为单个memmap缓存。

    使用v3文件名，避免错误读取先前版本生成的缓存。
    """

    cache_dir = Path(cfg.cache_dir)

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_prefix = (
        f"vepi_tcp20_"
        f"{cfg.epoch_seconds:g}s_"
        f"{cfg.target_len}points_v3"
    )

    data_path = (
        cache_dir
        / f"{cache_prefix}_data.npy"
    )

    label_path = (
        cache_dir
        / f"{cache_prefix}_labels.npy"
    )

    group_path = (
        cache_dir
        / f"{cache_prefix}_groups.npy"
    )

    if (
        not cfg.rebuild_vepi_cache
        and data_path.exists()
        and label_path.exists()
        and group_path.exists()
    ):
        print(
            f"加载VEPI缓存：{data_path}"
        )

        return CachedVEPIDataset(
            data_path,
            label_path,
            group_path,
        )

    records = []

    for label, class_name in enumerate(
        ("True-IED", "Non-IED")
    ):
        class_dir = (
            Path(cfg.vepi_data_dir)
            / class_name
        )

        files = sorted(
            Path(path)
            for path in glob.glob(
                str(class_dir / "*.npy")
            )
        )

        if not files:
            raise FileNotFoundError(
                f"类别目录为空或不存在：{class_dir}"
            )

        for path in files:
            match = re.search(
                cfg.vepi_group_regex,
                path.name,
            )

            if match is None:
                raise ValueError(
                    f"无法从文件名提取患者编号：{path.name}"
                )

            patient_group = (
                match.group(1)
                if match.groups()
                else match.group(0)
            )

            records.append(
                (
                    path,
                    label,
                    patient_group,
                )
            )

    print(
        f"开始生成VEPI统一缓存，"
        f"共{len(records)}个样本"
    )

    temporary_data_path = (
        cache_dir
        / f"{cache_prefix}_data.partial.npy"
    )

    data_memmap = np.lib.format.open_memmap(
        temporary_data_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(records),
            cfg.channels,
            cfg.target_len,
        ),
    )

    labels = np.empty(
        len(records),
        dtype=np.int64,
    )

    groups = []

    length_statistics = {
        "True-IED": {},
        "Non-IED": {},
    }

    try:
        for index, (
            path,
            label,
            patient_group,
        ) in enumerate(records):

            raw_data = np.load(
                path,
                allow_pickle=False,
            )

            if raw_data.ndim != 2:
                raise ValueError(
                    f"VEPI文件不是二维数组："
                    f"{path}，shape={raw_data.shape}"
                )

            # 记录原始时间长度。
            if raw_data.shape[0] in {
                19,
                29,
            }:
                raw_length = raw_data.shape[1]

            elif raw_data.shape[1] in {
                19,
                29,
            }:
                raw_length = raw_data.shape[0]

            else:
                raw_length = -1

            class_name = (
                "True-IED"
                if label == 0
                else "Non-IED"
            )

            length_statistics[class_name][
                raw_length
            ] = (
                length_statistics[
                    class_name
                ].get(raw_length, 0)
                + 1
            )

            try:
                converted_data = (
                    convert_vepi_to_tcp(
                        raw_data,
                        cfg,
                        tcp_montage,
                    )
                )

            except Exception as error:
                raise RuntimeError(
                    f"VEPI文件转换失败：{path}\n"
                    f"原始形状：{raw_data.shape}"
                ) from error

            data_memmap[index] = (
                converted_data
            )

            labels[index] = label
            groups.append(patient_group)

            if (
                (index + 1) % 1000 == 0
                or index + 1 == len(records)
            ):
                print(
                    f"VEPI缓存："
                    f"{index + 1}/{len(records)}"
                )

        data_memmap.flush()

    finally:
        del data_memmap

    os.replace(
        temporary_data_path,
        data_path,
    )

    np.save(
        label_path,
        labels,
        allow_pickle=False,
    )

    np.save(
        group_path,
        np.asarray(groups, dtype=str),
        allow_pickle=False,
    )

    print("\nVEPI原始长度统计：")

    for class_name in (
        "True-IED",
        "Non-IED",
    ):
        print(
            f"{class_name}: "
            f"{length_statistics[class_name]}"
        )

    return CachedVEPIDataset(
        data_path,
        label_path,
        group_path,
    )


def load_tuev_datasets(
    cfg: Config,
    tuev_helper,
):
    """
    使用原TUEV程序的数据读取代码。

    split_name使用cross4s前缀，避免读取原来的1秒缓存。
    """

    data_root = Path(
        cfg.tuev_data_root
    )

    official_train = (
        tuev_helper.TUEVEpochDataset(
            data_root / "train",
            "cross4s_train",
            cfg,
        )
    )

    official_eval = (
        tuev_helper.TUEVEpochDataset(
            data_root / "eval",
            "cross4s_eval",
            cfg,
        )
    )

    return (
        official_train,
        official_eval,
    )


def split_cost(
    current,
    target,
):
    class_error = np.square(
        (
            current - target
        )
        / np.maximum(target, 1.0)
    ).sum()

    current_sizes = current.sum(axis=1)
    target_sizes = target.sum(axis=1)

    size_error = np.square(
        (
            current_sizes
            - target_sizes
        )
        / np.maximum(
            target_sizes,
            1.0,
        )
    ).sum()

    return float(
        class_error
        + 0.25 * size_error
    )


def group_stratified_partition(
    labels,
    groups,
    ratios,
    seed,
    required_splits=None,
    attempts=512,
):
    """无额外依赖的患者级近似分层划分。"""

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    groups = np.asarray(groups)

    ratios = np.asarray(
        ratios,
        dtype=np.float64,
    )

    ratios = ratios / ratios.sum()

    number_of_splits = len(ratios)
    unique_groups = np.unique(groups)

    if len(unique_groups) < number_of_splits:
        raise ValueError(
            "患者组数量少于需要划分的集合数量"
        )

    group_counts = {
        group: np.bincount(
            labels[groups == group],
            minlength=2,
        ).astype(np.float64)
        for group in unique_groups
    }

    total_counts = np.bincount(
        labels,
        minlength=2,
    ).astype(np.float64)

    target = (
        ratios[:, None]
        * total_counts[None, :]
    )

    if required_splits is None:
        required_splits = tuple(
            range(number_of_splits)
        )

    best_assignment = None
    best_score = float("inf")

    for attempt in range(attempts):
        rng = np.random.default_rng(
            seed + attempt * 1009
        )

        ordered_groups = (
            unique_groups.copy()
        )

        rng.shuffle(ordered_groups)

        current = np.zeros_like(target)
        assignment = {}

        for position, group in enumerate(
            ordered_groups
        ):
            remaining_groups = (
                len(ordered_groups)
                - position
            )

            empty_splits = np.flatnonzero(
                current.sum(axis=1) == 0
            )

            if (
                len(empty_splits) > 0
                and remaining_groups
                == len(empty_splits)
            ):
                candidate_splits = (
                    empty_splits
                )
            else:
                candidate_splits = range(
                    number_of_splits
                )

            candidate_costs = []

            for split_index in candidate_splits:
                trial = current.copy()

                trial[int(split_index)] += (
                    group_counts[group]
                )

                candidate_costs.append(
                    (
                        split_cost(
                            trial,
                            target,
                        ),
                        int(split_index),
                    )
                )

            minimum_cost = min(
                cost
                for cost, _ in candidate_costs
            )

            tied_splits = [
                split_index
                for cost, split_index
                in candidate_costs
                if abs(
                    cost - minimum_cost
                ) < 1e-12
            ]

            selected_split = int(
                rng.choice(tied_splits)
            )

            current[selected_split] += (
                group_counts[group]
            )

            assignment[str(group)] = (
                selected_split
            )

        valid = True

        for split_index in required_splits:
            if (
                current[split_index].sum() == 0
                or np.any(
                    current[split_index] == 0
                )
            ):
                valid = False
                break

        if not valid:
            continue

        final_score = split_cost(
            current,
            target,
        )

        if final_score < best_score:
            best_score = final_score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError(
            "无法获得同时包含两类的患者级划分"
        )

    split_indices = [
        []
        for _ in range(number_of_splits)
    ]

    for sample_index, group in enumerate(
        groups
    ):
        split_index = best_assignment[
            str(group)
        ]

        split_indices[split_index].append(
            sample_index
        )

    patient_sets = [
        set(groups[indices])
        for indices in split_indices
    ]

    for first in range(number_of_splits):
        for second in range(
            first + 1,
            number_of_splits,
        ):
            if (
                patient_sets[first]
                & patient_sets[second]
            ):
                raise RuntimeError(
                    "不同集合间出现重复患者"
                )

    return split_indices


def describe_split(
    name,
    dataset,
    indices,
):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    class_counts = np.bincount(
        dataset.labels[indices],
        minlength=2,
    )

    patient_number = len(
        np.unique(
            dataset.groups[indices]
        )
    )

    sample_fraction = (
        len(indices)
        / len(dataset)
    )

    print(
        f"{name:<22} | "
        f"患者={patient_number:>4d} | "
        f"样本={len(indices):>6d} | "
        f"比例={sample_fraction:.4f} | "
        f"[IED, Non-IED]="
        f"{class_counts.tolist()}"
    )


def make_loader(
    dataset,
    indices,
    cfg,
    shuffle,
    seed,
):
    if indices is None:
        selected_dataset = dataset

    else:
        selected_dataset = Subset(
            dataset,
            list(indices),
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        selected_dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        generator=generator,
    )


def make_joint_loader(
    source_dataset,
    source_indices,
    target_dataset,
    target_indices,
    cfg,
    seed,
):
    source_subset = Subset(
        source_dataset,
        list(source_indices),
    )

    target_subset = Subset(
        target_dataset,
        list(target_indices),
    )

    joint_dataset = ConcatDataset(
        [
            source_subset,
            target_subset,
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    if cfg.balance_joint_domains:
        number_of_source = len(
            source_subset
        )

        number_of_target = len(
            target_subset
        )

        weights = torch.cat(
            [
                torch.full(
                    (number_of_source,),
                    0.5 / number_of_source,
                    dtype=torch.double,
                ),
                torch.full(
                    (number_of_target,),
                    0.5 / number_of_target,
                    dtype=torch.double,
                ),
            ]
        )

        sampler = WeightedRandomSampler(
            weights,
            num_samples=(
                number_of_source
                + number_of_target
            ),
            replacement=True,
            generator=generator,
        )

        return DataLoader(
            joint_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=(
                DEVICE.type == "cuda"
            ),
        )

    return DataLoader(
        joint_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        generator=generator,
    )


def confusion_matrix(
    y_true,
    y_pred,
):
    matrix = np.zeros(
        (2, 2),
        dtype=np.int64,
    )

    np.add.at(
        matrix,
        (
            np.asarray(y_true),
            np.asarray(y_pred),
        ),
        1,
    )

    return matrix


def metrics_from_cm(matrix):
    """类别0=True-IED，为阳性类。"""

    true_positive = matrix[0, 0]
    false_negative = matrix[0, 1]
    false_positive = matrix[1, 0]
    true_negative = matrix[1, 1]

    accuracy = (
        np.trace(matrix)
        / max(matrix.sum(), 1)
    )

    precision = (
        true_positive
        / max(
            true_positive
            + false_positive,
            1,
        )
    )

    recall = (
        true_positive
        / max(
            true_positive
            + false_negative,
            1,
        )
    )

    specificity = (
        true_negative
        / max(
            true_negative
            + false_positive,
            1,
        )
    )

    f1 = (
        2 * precision * recall
        / max(
            precision + recall,
            1e-12,
        )
    )

    return (
        float(accuracy),
        float(precision),
        float(recall),
        float(specificity),
        float(f1),
    )


def binary_auroc(
    y_true,
    probability,
):
    positive = (
        np.asarray(y_true) == 0
    ).astype(np.int64)

    probability = np.asarray(
        probability
    )

    number_of_positive = int(
        positive.sum()
    )

    number_of_negative = (
        len(positive)
        - number_of_positive
    )

    if (
        number_of_positive == 0
        or number_of_negative == 0
    ):
        return float("nan")

    order = np.argsort(
        -probability,
        kind="mergesort",
    )

    sorted_labels = positive[order]
    sorted_scores = probability[order]

    distinct_indices = np.r_[
        np.flatnonzero(
            np.diff(sorted_scores)
        ),
        len(sorted_scores) - 1,
    ]

    true_positive = np.cumsum(
        sorted_labels
    )[distinct_indices]

    false_positive = (
        distinct_indices
        + 1
        - true_positive
    )

    true_positive_rate = np.r_[
        0.0,
        true_positive / number_of_positive,
        1.0,
    ]

    false_positive_rate = np.r_[
        0.0,
        false_positive / number_of_negative,
        1.0,
    ]

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(
                true_positive_rate,
                false_positive_rate,
            )
        )

    return float(
        np.trapz(
            true_positive_rate,
            false_positive_rate,
        )
    )


def average_precision(
    y_true,
    probability,
):
    positive = (
        np.asarray(y_true) == 0
    ).astype(np.int64)

    number_of_positive = int(
        positive.sum()
    )

    if number_of_positive == 0:
        return float("nan")

    order = np.argsort(
        -np.asarray(probability),
        kind="mergesort",
    )

    sorted_labels = positive[order]

    cumulative_positive = np.cumsum(
        sorted_labels
    )

    precision = (
        cumulative_positive
        / np.arange(
            1,
            len(sorted_labels) + 1,
        )
    )

    return float(
        (
            precision
            * sorted_labels
        ).sum()
        / number_of_positive
    )


@torch.inference_mode()
def predict(
    model,
    loader,
):
    model.eval()

    labels = []
    probabilities = []

    for data, target in loader:
        data = data.to(
            DEVICE,
            non_blocking=True,
        )

        logits, _ = model(data)

        ied_probability = (
            logits.float()
            .softmax(dim=1)[:, 0]
            .cpu()
            .numpy()
        )

        probabilities.extend(
            ied_probability
        )

        labels.extend(
            target.numpy()
        )

    return (
        np.asarray(labels),
        np.asarray(probabilities),
    )


def evaluate(
    model,
    loader,
    threshold,
):
    labels, probabilities = predict(
        model,
        loader,
    )

    predictions = np.where(
        probabilities >= threshold,
        0,
        1,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    (
        accuracy,
        precision,
        recall,
        specificity,
        f1,
    ) = metrics_from_cm(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": binary_auroc(
            labels,
            probabilities,
        ),
        "auprc": average_precision(
            labels,
            probabilities,
        ),
        "threshold": float(threshold),
        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def build_model(cfg):
    return CausalIEDGraphNet(
        cfg
    ).to(DEVICE)


def state_to_cpu(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter
        in model.state_dict().items()
    }


def fit_stage(
    model,
    train_loader,
    val_loader,
    epochs,
    patience,
    learning_rate,
    cfg,
    stage_name,
):
    criterion = nn.CrossEntropyLoss().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.10,
            div_factor=10.0,
            final_div_factor=100.0,
        )
    )

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        loss_sum = 0.0

        for data, target in train_loader:
            data = data.to(
                DEVICE,
                non_blocking=True,
            )

            target = target.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, _ = model(data)

            loss = criterion(
                logits,
                target,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += float(
                loss.detach()
            )

        validation_result = evaluate(
            model,
            val_loader,
            cfg.fixed_threshold,
        )

        validation_score = (
            validation_result["auprc"]
        )

        if (
            np.isfinite(validation_score)
            and validation_score > best_score
        ):
            best_score = validation_score
            best_state = state_to_cpu(model)
            best_epoch = epoch
            wait = 0

        else:
            wait += 1

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"{stage_name} | "
                f"Epoch {epoch:03d} | "
                f"loss "
                f"{loss_sum / max(len(train_loader), 1):.4f} | "
                f"val AUROC "
                f"{validation_result['auroc']:.4f} | "
                f"val AUPRC "
                f"{validation_result['auprc']:.4f} | "
                f"ACC "
                f"{validation_result['accuracy']:.4f} | "
                f"F1 "
                f"{validation_result['f1']:.4f} | "
                f"patience {wait}/{patience}"
            )

        if wait >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"{stage_name}未获得有效模型"
        )

    model.load_state_dict(
        best_state
    )

    information = {
        "best_epoch": best_epoch,
        "best_val_auprc": float(
            best_score
        ),
    }

    return (
        best_state,
        information,
    )


def print_result(
    seed,
    protocol,
    result,
):
    print(
        f"Seed {seed:>3d} | "
        f"{protocol:<19} | "
        f"ACC "
        f"{result['accuracy'] * 100:.2f}% | "
        f"PREC "
        f"{result['precision'] * 100:.2f}% | "
        f"RECALL "
        f"{result['recall'] * 100:.2f}% | "
        f"SPEC "
        f"{result['specificity'] * 100:.2f}% | "
        f"F1 "
        f"{result['f1'] * 100:.2f}% | "
        f"AUROC "
        f"{result['auroc'] * 100:.2f}% | "
        f"AUPRC "
        f"{result['auprc'] * 100:.2f}%"
    )


def create_result_row(
    seed,
    protocol,
    result,
    pretrain_information=None,
    target_information=None,
):
    row = {
        "seed": seed,
        "protocol": protocol,
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "specificity": result["specificity"],
        "f1": result["f1"],
        "auroc": result["auroc"],
        "auprc": result["auprc"],
        "threshold": result["threshold"],
        "confusion_matrix": json.dumps(
            result["confusion_matrix"]
        ),
        "pretrain_best_epoch": "",
        "pretrain_best_val_auprc": "",
        "target_best_epoch": "",
        "target_best_val_auprc": "",
    }

    if pretrain_information is not None:
        row["pretrain_best_epoch"] = (
            pretrain_information[
                "best_epoch"
            ]
        )

        row["pretrain_best_val_auprc"] = (
            pretrain_information[
                "best_val_auprc"
            ]
        )

    if target_information is not None:
        row["target_best_epoch"] = (
            target_information[
                "best_epoch"
            ]
        )

        row["target_best_val_auprc"] = (
            target_information[
                "best_val_auprc"
            ]
        )

    return row


def write_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    cfg,
    seed,
    protocol,
    state,
):
    if not cfg.save_checkpoints:
        return

    checkpoint_dir = (
        Path(cfg.output_dir)
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_protocol = (
        protocol.lower()
        .replace("-", "_")
    )

    torch.save(
        {
            "seed": seed,
            "protocol": protocol,
            "model_state_dict": state,
        },
        checkpoint_dir
        / f"{safe_protocol}_seed_{seed}.pt",
    )


def run_one_seed(
    cfg,
    seed,
    vepi_dataset,
    tuev_train_dataset,
    tuev_eval_dataset,
    vepi_train_indices,
    vepi_val_indices,
    target_train_indices,
    target_val_indices,
):
    print("\n" + "=" * 85)
    print(
        f"Seed={seed}, Device={DEVICE}"
    )

    rows = []

    source_val_loader = make_loader(
        vepi_dataset,
        vepi_val_indices,
        cfg,
        shuffle=False,
        seed=seed + 1,
    )

    target_val_loader = make_loader(
        tuev_train_dataset,
        target_val_indices,
        cfg,
        shuffle=False,
        seed=seed + 2,
    )

    target_eval_loader = make_loader(
        tuev_eval_dataset,
        None,
        cfg,
        shuffle=False,
        seed=seed + 3,
    )

    # ==========================================================
    # VEPI预训练
    # ==========================================================
    set_seed(seed)

    source_model = build_model(cfg)

    source_train_loader = make_loader(
        vepi_dataset,
        vepi_train_indices,
        cfg,
        shuffle=True,
        seed=seed + 10,
    )

    (
        source_state,
        pretrain_information,
    ) = fit_stage(
        source_model,
        source_train_loader,
        source_val_loader,
        cfg.pretrain_epochs,
        cfg.pretrain_patience,
        cfg.pretrain_lr,
        cfg,
        "VEPI pretraining",
    )

    # ==========================================================
    # Source-Only
    # ==========================================================
    source_result = evaluate(
        source_model,
        target_eval_loader,
        cfg.fixed_threshold,
    )

    print_result(
        seed,
        "Source-Only",
        source_result,
    )

    rows.append(
        create_result_row(
            seed,
            "Source-Only",
            source_result,
            pretrain_information=(
                pretrain_information
            ),
        )
    )

    save_checkpoint(
        cfg,
        seed,
        "Source-Only",
        source_state,
    )

    # ==========================================================
    # Pretrain-Finetune
    # ==========================================================
    set_seed(seed)

    finetune_model = build_model(cfg)

    finetune_model.load_state_dict(
        source_state
    )

    finetune_loader = make_loader(
        tuev_train_dataset,
        target_train_indices,
        cfg,
        shuffle=True,
        seed=seed + 20,
    )

    (
        finetune_state,
        finetune_information,
    ) = fit_stage(
        finetune_model,
        finetune_loader,
        target_val_loader,
        cfg.finetune_epochs,
        cfg.finetune_patience,
        cfg.finetune_lr,
        cfg,
        "TUEV fine-tuning",
    )

    finetune_result = evaluate(
        finetune_model,
        target_eval_loader,
        cfg.fixed_threshold,
    )

    print_result(
        seed,
        "Pretrain-Finetune",
        finetune_result,
    )

    rows.append(
        create_result_row(
            seed,
            "Pretrain-Finetune",
            finetune_result,
            pretrain_information=(
                pretrain_information
            ),
            target_information=(
                finetune_information
            ),
        )
    )

    save_checkpoint(
        cfg,
        seed,
        "Pretrain-Finetune",
        finetune_state,
    )

    # ==========================================================
    # Target-Only
    # ==========================================================
    set_seed(seed)

    target_model = build_model(cfg)

    target_train_loader = make_loader(
        tuev_train_dataset,
        target_train_indices,
        cfg,
        shuffle=True,
        seed=seed + 30,
    )

    (
        target_state,
        target_information,
    ) = fit_stage(
        target_model,
        target_train_loader,
        target_val_loader,
        cfg.target_epochs,
        cfg.target_patience,
        cfg.target_lr,
        cfg,
        "TUEV target-only",
    )

    target_result = evaluate(
        target_model,
        target_eval_loader,
        cfg.fixed_threshold,
    )

    print_result(
        seed,
        "Target-Only",
        target_result,
    )

    rows.append(
        create_result_row(
            seed,
            "Target-Only",
            target_result,
            target_information=(
                target_information
            ),
        )
    )

    save_checkpoint(
        cfg,
        seed,
        "Target-Only",
        target_state,
    )

    # ==========================================================
    # Joint-Training
    # ==========================================================
    set_seed(seed)

    joint_model = build_model(cfg)

    joint_loader = make_joint_loader(
        vepi_dataset,
        vepi_train_indices,
        tuev_train_dataset,
        target_train_indices,
        cfg,
        seed=seed + 40,
    )

    (
        joint_state,
        joint_information,
    ) = fit_stage(
        joint_model,
        joint_loader,
        target_val_loader,
        cfg.joint_epochs,
        cfg.joint_patience,
        cfg.joint_lr,
        cfg,
        "Joint training",
    )

    joint_result = evaluate(
        joint_model,
        target_eval_loader,
        cfg.fixed_threshold,
    )

    print_result(
        seed,
        "Joint-Training",
        joint_result,
    )

    rows.append(
        create_result_row(
            seed,
            "Joint-Training",
            joint_result,
            target_information=(
                joint_information
            ),
        )
    )

    save_checkpoint(
        cfg,
        seed,
        "Joint-Training",
        joint_state,
    )

    del (
        source_model,
        finetune_model,
        target_model,
        joint_model,
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return rows


def summarize_results(rows):
    protocol_order = (
        "Pretrain-Finetune",
        "Source-Only",
        "Target-Only",
        "Joint-Training",
    )

    summary_rows = []

    print(
        "\n"
        + "=" * 24
        + " Final summary "
        + "=" * 24
    )

    for protocol in protocol_order:
        selected_rows = [
            row
            for row in rows
            if row["protocol"] == protocol
        ]

        if not selected_rows:
            continue

        summary_row = {
            "protocol": protocol,
            "n_seeds": len(selected_rows),
        }

        display_parts = []

        for metric_name in METRIC_NAMES:
            values = np.asarray(
                [
                    row[metric_name]
                    for row in selected_rows
                ],
                dtype=np.float64,
            )

            mean = float(
                np.nanmean(values)
            )

            std = (
                float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0
            )

            summary_row[
                f"{metric_name}_mean"
            ] = mean

            summary_row[
                f"{metric_name}_std"
            ] = std

            display_parts.append(
                f"{metric_name.upper()} "
                f"{mean * 100:.2f} ± "
                f"{std * 100:.2f}%"
            )

        summary_rows.append(
            summary_row
        )

        print(f"\n{protocol}")
        print(" | ".join(display_parts))

    return summary_rows


def validate_config(
    cfg,
    tcp_montage,
):
    if cfg.channels != len(tcp_montage):
        raise ValueError(
            f"模型通道数为{cfg.channels}，"
            f"TCP导联数为{len(tcp_montage)}"
        )

    if (
        cfg.target_len != 2000
        or cfg.epoch_seconds != 4.0
    ):
        print(
            "警告：当前公共输入已不再是"
            "4秒、500 Hz和2000点"
        )

    if not (
        0 < cfg.tuev_train_fraction < 1
    ):
        raise ValueError(
            "tuev_train_fraction必须位于(0,1)"
        )

    if not (
        0 < cfg.target_val_fraction < 1
    ):
        raise ValueError(
            "target_val_fraction必须位于(0,1)"
        )

    if (
        cfg.tuev_train_fraction
        + cfg.target_val_fraction
        >= 1
    ):
        raise ValueError(
            "TUEV训练和验证比例之和必须小于1"
        )

    if not (
        0 < cfg.source_val_fraction < 1
    ):
        raise ValueError(
            "source_val_fraction必须位于(0,1)"
        )

    if (
        len(cfg.seeds) != 10
        or len(set(cfg.seeds)) != 10
    ):
        raise ValueError(
            "正式实验必须包含10个不同随机种子"
        )


def main():
    cfg = Config()

    script_dir = Path(
        __file__
    ).resolve().parent

    if str(script_dir) not in sys.path:
        sys.path.insert(
            0,
            str(script_dir),
        )

    helper_path = Path(
        cfg.tuev_helper_script
    )

    if not helper_path.is_absolute():
        helper_path = (
            script_dir
            / helper_path
        )

    tuev_helper = load_tuev_helper(
        helper_path
    )

    validate_config(
        cfg,
        tuev_helper.TCP_MONTAGE_20,
    )

    output_dir = Path(
        cfg.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        output_dir
        / "cross_dataset_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(cfg),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Device: {DEVICE}")

    print(
        f"统一输入："
        f"{cfg.channels}通道TCP，"
        f"{cfg.epoch_seconds:g}秒，"
        f"{cfg.target_len}点"
    )

    print(
        "VEPI通道顺序："
        f"{list(cfg.vepi_channel_names)}"
    )

    print(
        f"IED固定分类阈值："
        f"{cfg.fixed_threshold:.3f}"
    )

    # ==========================================================
    # 数据加载及公共预处理
    # ==========================================================
    vepi_dataset = load_or_build_vepi(
        cfg,
        tuev_helper.TCP_MONTAGE_20,
    )

    (
        tuev_train_dataset,
        tuev_eval_dataset,
    ) = load_tuev_datasets(
        cfg,
        tuev_helper,
    )

    print(
        f"\nVEPI样本数：{len(vepi_dataset)}"
    )

    print(
        f"TUEV官方train样本数："
        f"{len(tuev_train_dataset)}"
    )

    print(
        f"TUEV官方evaluation样本数："
        f"{len(tuev_eval_dataset)}"
    )

    # ==========================================================
    # 固定患者级划分
    # ==========================================================
    (
        vepi_train_indices,
        vepi_val_indices,
    ) = group_stratified_partition(
        vepi_dataset.labels,
        vepi_dataset.groups,
        ratios=(
            1.0 - cfg.source_val_fraction,
            cfg.source_val_fraction,
        ),
        seed=cfg.split_seed,
        required_splits=(0, 1),
    )

    unused_fraction = (
        1.0
        - cfg.tuev_train_fraction
        - cfg.target_val_fraction
    )

    (
        target_train_indices,
        target_val_indices,
        unused_indices,
    ) = group_stratified_partition(
        tuev_train_dataset.labels,
        tuev_train_dataset.groups,
        ratios=(
            cfg.tuev_train_fraction,
            cfg.target_val_fraction,
            unused_fraction,
        ),
        seed=cfg.split_seed,
        required_splits=(0, 1),
    )

    print("\n固定患者级数据划分：")

    describe_split(
        "VEPI train",
        vepi_dataset,
        vepi_train_indices,
    )

    describe_split(
        "VEPI validation",
        vepi_dataset,
        vepi_val_indices,
    )

    describe_split(
        "TUEV 10% train",
        tuev_train_dataset,
        target_train_indices,
    )

    describe_split(
        "TUEV validation",
        tuev_train_dataset,
        target_val_indices,
    )

    describe_split(
        "TUEV unused",
        tuev_train_dataset,
        unused_indices,
    )

    describe_split(
        "TUEV official eval",
        tuev_eval_dataset,
        list(
            range(
                len(tuev_eval_dataset)
            )
        ),
    )

    # ==========================================================
    # 模型接口检查
    # ==========================================================
    test_model = build_model(cfg)

    parameter_number = sum(
        parameter.numel()
        for parameter in test_model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\n模型可训练参数量："
        f"{parameter_number:,}"
    )

    with torch.inference_mode():
        dummy_input = torch.zeros(
            2,
            cfg.channels,
            cfg.target_len,
            device=DEVICE,
        )

        logits, adjacency = test_model(
            dummy_input
        )

    expected_adjacency_shape = (
        2,
        cfg.channels,
        cfg.channels,
    )

    if (
        logits.shape != (2, 2)
        or adjacency.shape
        != expected_adjacency_shape
    ):
        raise RuntimeError(
            f"模型输出异常："
            f"logits={logits.shape}，"
            f"adjacency={adjacency.shape}"
        )

    del (
        test_model,
        dummy_input,
        logits,
        adjacency,
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ==========================================================
    # 10个训练随机种子
    # ==========================================================
    all_results = []

    per_seed_path = (
        output_dir
        / "cross_dataset_per_seed.csv"
    )

    for run_index, seed in enumerate(
        cfg.seeds,
        start=1,
    ):
        print(
            f"\n开始第"
            f"{run_index}/{len(cfg.seeds)}"
            f"次实验"
        )

        seed_results = run_one_seed(
            cfg,
            int(seed),
            vepi_dataset,
            tuev_train_dataset,
            tuev_eval_dataset,
            vepi_train_indices,
            vepi_val_indices,
            target_train_indices,
            target_val_indices,
        )

        all_results.extend(
            seed_results
        )

        # 每个seed结束后立即保存。
        write_csv(
            per_seed_path,
            all_results,
        )

    summary_results = summarize_results(
        all_results
    )

    summary_path = (
        output_dir
        / "cross_dataset_summary.csv"
    )

    write_csv(
        summary_path,
        summary_results,
    )

    print(
        f"\n逐seed结果：{per_seed_path}"
    )

    print(
        f"均值和标准差：{summary_path}"
    )


if __name__ == "__main__":
    main()