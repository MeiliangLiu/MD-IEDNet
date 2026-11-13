import copy
import csv
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset, Subset

# from causal_ied_graph import CausalIEDGraphNet
from model import CausalIEDGraphNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEEDS = tuple(range(42,52))
# RANDOM_SEEDS = tuple([43, 44, 45, 46, 47])

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # 目录下应包含 train 和 eval 两个文件夹，EDF与同名REC位于其各级子目录中。
    data_root: str = r"D:\TUEV\v2.0.1\edf"
    cache_dir: str = r"D:\TUEV\v2.0.1\epoch_cache"
    rebuild_cache: bool = False

    # TUEV epoch-level设置：20通道TCP、1秒、重采样至256 Hz。
    target_len: int = 256
    channels: int = 20
    epoch_seconds: float = 1.0
    time_bins: int = 16
    batch_size: int = 96
    epochs: int = 100
    patience: int = 15

    lr: float = 1e-3
    dropout: float = 0.1

    weight_decay: float = 1e-4
    node_dim: int = 24
    graph_hidden: int = 48
    rank_dim: int = 12

    # 仅从官方train中按患者文件夹划出验证集；官方eval始终作为最终测试集。
    val_ratio: float = 0.1
    # 固定患者级train/val划分，使10次实验只改变模型训练随机性。
    split_seed: int = 42
    seed: int = 42
    compile_model: bool = False
    checkpoint_path: str = "best_tuev_epoch_causal_ied_graphnet.pt"


LABEL_NAMES = {
    1: "SPSW",
    2: "GPED",
    3: "PLED",
    4: "EYEM",
    5: "ARTF",
    6: "BCKG",
}
IED_LABELS = {1, 2, 3}

# 标准22通道TCP中去除依赖A1/A2的A1-T3和T4-A2，保留20个双极导联。
TCP_MONTAGE_20 = (
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T4"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)

# 新旧10-20命名兼容。
ELECTRODE_ALIASES = {
    "T3": ("T3", "T7"),
    "T4": ("T4", "T8"),
    "T5": ("T5", "P7"),
    "T6": ("T6", "P8"),
}


class SimpleEDFReader:
    """读取TUEV标准EDF，无需额外安装MNE或pyEDFlib。"""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as file:
            fixed = file.read(256)

            def fixed_field(start, stop):
                return fixed[start:stop].decode("ascii", "replace").strip()

            self.header_bytes = int(fixed_field(184, 192))
            self.n_records = int(fixed_field(236, 244))
            self.record_duration = float(fixed_field(244, 252))
            self.n_signals = int(fixed_field(252, 256))
            signal_header = file.read(self.header_bytes - 256)

        widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
        fields, position = [], 0
        for width in widths:
            values = []
            for index in range(self.n_signals):
                begin = position + index * width
                values.append(
                    signal_header[begin : begin + width]
                    .decode("ascii", "replace")
                    .strip()
                )
            fields.append(values)
            position += self.n_signals * width

        self.ch_names = fields[0]
        self.physical_min = np.asarray(fields[3], dtype=np.float64)
        self.physical_max = np.asarray(fields[4], dtype=np.float64)
        self.digital_min = np.asarray(fields[5], dtype=np.float64)
        self.digital_max = np.asarray(fields[6], dtype=np.float64)
        self.samples_per_record = np.asarray(fields[8], dtype=np.int64)
        self.record_offsets = np.r_[0, np.cumsum(self.samples_per_record)[:-1]]
        self.samples_in_record = int(self.samples_per_record.sum())

        if self.n_records <= 0 or self.record_duration <= 0:
            raise ValueError(f"EDF记录参数无效: {self.path}")
        self._memmap = np.memmap(
            self.path,
            mode="r",
            dtype="<i2",
            offset=self.header_bytes,
            shape=(self.n_records, self.samples_in_record),
        )
        self._cache = {}
        self.info = {}
        self.n_times = None

    def sampling_frequency(self, signal_index):
        return float(self.samples_per_record[signal_index] / self.record_duration)

    def configure_reference_frequency(self, signal_indices):
        frequencies = [self.sampling_frequency(index) for index in signal_indices]
        if not np.allclose(frequencies, frequencies[0]):
            raise ValueError(f"所需EEG通道采样率不一致: {frequencies}")
        self.info["sfreq"] = frequencies[0]
        self.n_times = int(self.n_records * self.samples_per_record[signal_indices[0]])

    def _load_signal(self, signal_index):
        if signal_index not in self._cache:
            offset = int(self.record_offsets[signal_index])
            count = int(self.samples_per_record[signal_index])
            digital = np.asarray(
                self._memmap[:, offset : offset + count], dtype=np.float32
            ).reshape(-1)
            denominator = self.digital_max[signal_index] - self.digital_min[signal_index]
            if denominator == 0:
                raise ValueError(f"EDF数字量范围为0: {self.ch_names[signal_index]}")
            scale = (
                self.physical_max[signal_index] - self.physical_min[signal_index]
            ) / denominator
            physical = (
                (digital - self.digital_min[signal_index]) * scale
                + self.physical_min[signal_index]
            )
            self._cache[signal_index] = np.asarray(physical, dtype=np.float32)
        return self._cache[signal_index]

    def get_data(self, picks, start, stop):
        return np.stack(
            [self._load_signal(index)[start:stop] for index in picks], axis=0
        )

    def close(self):
        self._cache.clear()
        del self._memmap


def normalize_channel_name(name: str) -> str:
    name = name.upper().strip()
    if name.startswith("EEG "):
        name = name[4:]
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def read_rec_epochs(rec_path: Path):
    """按相同起止时间合并通道标注，返回(start, stop, binary_label)。"""
    grouped = {}
    with rec_path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_no, row in enumerate(csv.reader(file), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 4:
                raise ValueError(f"{rec_path} 第{line_no}行不足4列")
            start = float(row[1])
            stop = float(row[2])
            event_label = int(float(row[3]))
            if event_label not in LABEL_NAMES:
                raise ValueError(f"{rec_path} 第{line_no}行标签未知: {event_label}")
            key = (round(start, 6), round(stop, 6))
            grouped.setdefault(key, set()).add(event_label)

    epochs = []
    for (start, stop), event_labels in sorted(grouped.items()):
        # 保持原模型的类别约定：0=True-IED，1=Non-IED。
        binary_label = 0 if event_labels & IED_LABELS else 1
        epochs.append((start, stop, binary_label))
    return epochs


def resolve_electrodes(raw):
    normalized_to_index = {
        normalize_channel_name(name): index for index, name in enumerate(raw.ch_names)
    }
    required = sorted({electrode for pair in TCP_MONTAGE_20 for electrode in pair})
    resolved = {}
    for electrode in required:
        candidates = ELECTRODE_ALIASES.get(electrode, (electrode,))
        match = next(
            (normalized_to_index[name] for name in candidates if name in normalized_to_index),
            None,
        )
        if match is None:
            raise ValueError(
                f"缺少电极{electrode}；现有通道为: {', '.join(raw.ch_names)}"
            )
        resolved[electrode] = match
    raw.configure_reference_frequency(list(resolved.values()))
    return resolved


def extract_epoch(raw, electrode_indices, start, stop, cfg: Config):
    """从单极EDF提取固定1秒信号，转换为20通道TCP并重采样。"""
    sfreq = float(raw.info["sfreq"])
    source_len = max(1, int(round(cfg.epoch_seconds * sfreq)))
    center = 0.5 * (start + stop)
    sample_start = int(round(center * sfreq)) - source_len // 2
    sample_stop = sample_start + source_len

    left_pad = max(0, -sample_start)
    right_pad = max(0, sample_stop - raw.n_times)
    read_start = max(0, sample_start)
    read_stop = min(raw.n_times, sample_stop)

    electrodes = list(electrode_indices)
    picks = [electrode_indices[name] for name in electrodes]
    data = raw.get_data(picks=picks, start=read_start, stop=read_stop)
    if left_pad or right_pad:
        data = np.pad(data, ((0, 0), (left_pad, right_pad)), mode="edge")
    if data.shape[1] != source_len:
        raise RuntimeError(f"epoch长度异常: 期望{source_len}，实际{data.shape[1]}")

    signals = {name: data[index] for index, name in enumerate(electrodes)}
    tcp = np.stack([signals[left] - signals[right] for left, right in TCP_MONTAGE_20])
    if tcp.shape[1] != cfg.target_len:
        tcp = resample(tcp, cfg.target_len, axis=1)

    tcp = np.asarray(tcp, dtype=np.float32)
    tcp = (tcp - tcp.mean(axis=1, keepdims=True)) / np.maximum(
        tcp.std(axis=1, keepdims=True), 1e-6
    )
    return np.ascontiguousarray(tcp)


class TUEVEpochDataset(Dataset):
    classes = ("True-IED", "Non-IED")

    def __init__(self, split_dir: str, split_name: str, cfg: Config):
        self.split_dir = Path(split_dir)
        cache_dir = Path(cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"tuev_{split_name}_epoch20x{cfg.target_len}.npz"

        if cache_path.exists() and not cfg.rebuild_cache:
            with np.load(cache_path, allow_pickle=False) as cached:
                self.data = cached["data"]
                self.labels = cached["labels"]
                self.groups = cached["groups"]
            print(f"已从缓存加载 {split_name}: {cache_path}")
        else:
            self.data, self.labels, self.groups = self._build(cfg)
            np.savez_compressed(
                cache_path,
                data=self.data,
                labels=self.labels,
                groups=self.groups,
            )
            print(f"已保存 {split_name} 缓存: {cache_path}")

        if self.data.ndim != 3 or self.data.shape[1:] != (
            cfg.channels,
            cfg.target_len,
        ):
            raise ValueError(f"{split_name}缓存形状错误: {self.data.shape}")
        print(
            f"{split_name}: {len(self.labels)} epochs, 形状={self.data.shape}, "
            f"[IED, Non-IED]={np.bincount(self.labels, minlength=2).tolist()}"
        )

    def _build(self, cfg: Config):
        rec_files = sorted(self.split_dir.rglob("*.rec"))
        if not rec_files:
            raise FileNotFoundError(f"没有在{self.split_dir}下找到REC文件")

        arrays, labels, groups = [], [], []
        failures = []
        for file_index, rec_path in enumerate(rec_files, start=1):
            edf_path = rec_path.with_suffix(".edf")
            if not edf_path.exists():
                failures.append(f"缺少同名EDF: {rec_path}")
                continue
            raw = None
            try:
                epochs = read_rec_epochs(rec_path)
                if not epochs:
                    continue
                raw = SimpleEDFReader(edf_path)
                electrode_indices = resolve_electrodes(raw)
                relative = rec_path.relative_to(self.split_dir)
                # 第一层子文件夹为患者组；若文件直接位于根目录，则以文件名为组。
                patient_group = relative.parts[0] if len(relative.parts) > 1 else rec_path.stem
                for start, stop, label in epochs:
                    arrays.append(extract_epoch(raw, electrode_indices, start, stop, cfg))
                    labels.append(label)
                    groups.append(patient_group)
            except Exception as exc:
                failures.append(f"{rec_path}: {exc}")
            finally:
                if raw is not None:
                    raw.close()

            if file_index % 25 == 0 or file_index == len(rec_files):
                print(
                    f"处理 {self.split_dir.name}: {file_index}/{len(rec_files)} REC, "
                    f"已生成{len(labels)}个epochs"
                )

        if failures:
            preview = "\n".join(failures[:10])
            raise RuntimeError(
                f"共有{len(failures)}个文件预处理失败。前10项：\n{preview}"
            )
        if not arrays:
            raise RuntimeError(f"{self.split_dir}未生成任何epoch")
        return (
            np.stack(arrays).astype(np.float32, copy=False),
            np.asarray(labels, dtype=np.int64),
            np.asarray(groups),
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), int(self.labels[index])


def group_stratified_train_val_indices(labels, groups, val_ratio, seed):
    """在官方train内按患者文件夹进行近似分层的train/val划分。"""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    if len(unique_groups) < 2:
        raise ValueError("患者组少于2组，无法划分train/val")

    group_counts = {
        group: np.bincount(labels[groups == group], minlength=2).astype(np.float64)
        for group in unique_groups
    }
    total = np.bincount(labels, minlength=2).astype(np.float64)
    target = np.stack([total * (1.0 - val_ratio), total * val_ratio])
    current = np.zeros_like(target)
    assignment = {}

    # 大组优先；同样大小的组保留上面随机打乱后的次序。
    order_rank = {group: i for i, group in enumerate(unique_groups)}
    ordered = sorted(
        unique_groups,
        key=lambda group: (-group_counts[group].sum(), order_rank[group]),
    )
    for group in ordered:
        count = group_counts[group]
        costs = []
        for split in range(2):
            trial = current.copy()
            trial[split] += count
            class_error = np.square((trial - target) / np.maximum(target, 1.0)).sum()
            size_error = np.square(
                (trial.sum(1) - target.sum(1)) / np.maximum(target.sum(1), 1.0)
            ).sum()
            costs.append(class_error + 0.25 * size_error)
        chosen = int(np.argmin(costs))
        current[chosen] += count
        assignment[group] = chosen

    indices = [[], []]
    for idx, group in enumerate(groups):
        indices[assignment[group]].append(idx)
    if any(len(split) == 0 for split in indices):
        raise RuntimeError("患者组分布不适合当前比例，某个数据集为空")
    return indices[0], indices[1]


def make_train_val_split(dataset: TUEVEpochDataset, cfg: Config):
    splits = group_stratified_train_val_indices(
        dataset.labels, dataset.groups, cfg.val_ratio, cfg.split_seed
    )
    group_sets = [set(dataset.groups[idx]) for idx in splits]
    if group_sets[0] & group_sets[1]:
        raise RuntimeError("患者级划分失败：train与val出现重复患者")
    print(f"官方train内患者组数量 train/val: {[len(x) for x in group_sets]}")
    return splits


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def metrics_from_cm(cm):
    """以类别0（True-IED）为阳性类计算二分类指标。"""
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]

    accuracy = np.trace(cm) / max(cm.sum(), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return (
        float(accuracy),
        float(precision),
        float(recall),
        float(specificity),
        float(f1),
    )


def binary_auroc(y_true, probability):
    """类别0(True-IED)作为阳性类；使用梯形积分，处理重复概率。"""
    positive = (np.asarray(y_true) == 0).astype(np.int64)
    probability = np.asarray(probability)
    n_pos, n_neg = positive.sum(), len(positive) - positive.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    y = positive[order]
    scores = probability[order]
    distinct = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    tpr = np.r_[0.0, tp / n_pos, 1.0]
    fpr = np.r_[0.0, fp / n_neg, 1.0]
    return float(np.trapz(tpr, fpr))


def average_precision(y_true, probability):
    """非插值Average Precision，类别0(True-IED)作为阳性类。"""
    positive = (np.asarray(y_true) == 0).astype(np.int64)
    n_pos = positive.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probability), kind="mergesort")
    y = positive[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / n_pos)


def threshold_metrics(y_true, ied_probability, threshold):
    """计算指定IED概率阈值下的结果；类别0为阳性IED。"""
    pred = np.where(ied_probability >= threshold, 0, 1)
    cm = confusion_matrix(y_true, pred)
    acc, precision, recall, specificity, f1 = metrics_from_cm(cm)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": cm,
    }


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits, _ = model(x)
        probabilities.extend(logits.float().softmax(dim=1)[:, 0].cpu().numpy())
        labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def make_loaders(train_dataset, eval_dataset, train_idx, val_idx, cfg):
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    return (
        DataLoader(
            Subset(train_dataset, train_idx),
            shuffle=True,
            drop_last=False,
            **common,
        ),
        DataLoader(Subset(train_dataset, val_idx), shuffle=False, **common),
        DataLoader(eval_dataset, shuffle=False, **common),
    )


def train(cfg: Config):
    print(f"\n{'=' * 24} Seed {cfg.seed} {'=' * 24}")
    set_seed(cfg.seed)
    data_root = Path(cfg.data_root)
    official_train = TUEVEpochDataset(
        data_root / "train", "train", cfg
    )
    official_eval = TUEVEpochDataset(
        data_root / "eval", "eval", cfg
    )
    train_idx, val_idx = make_train_val_split(official_train, cfg)
    train_loader, val_loader, test_loader = make_loaders(
        official_train, official_eval, train_idx, val_idx, cfg
    )
    train_counts = np.bincount(official_train.labels[train_idx], minlength=2)
    print(
        f"train/val/official-eval: {len(train_idx)}/{len(val_idx)}/{len(official_eval)}"
    )
    print(f"训练集类别计数 [True-IED, Non-IED]: {train_counts.tolist()}")

    # cfg.graph_variant = "topology"
    # cfg.graph_variant = "topology_salient"
    base_model = CausalIEDGraphNet(cfg).to(DEVICE)
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"模型可训练参数量: {trainable_params:,}")
    model = base_model
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(base_model)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.10,
        div_factor=10.0,
        final_div_factor=100.0,
    )
    # AUPRC不依赖阈值，适合不平衡IED任务，也避免每个epoch反复拟合验证阈值。
    best_score, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())

        val_y, val_prob = predict(model, val_loader)
        val_auc = binary_auroc(val_y, val_prob)
        val_ap = average_precision(val_y, val_prob)
        fixed_pred = np.where(val_prob >= 0.5, 0, 1)
        val_acc, val_precision, val_recall, val_specificity, val_f1 = metrics_from_cm(
            confusion_matrix(val_y, fixed_pred)
        )

        if val_ap > best_score:
            best_score = val_ap
            best_state = copy.deepcopy(base_model.state_dict())
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            batches = len(train_loader)
            print(
                f"Epoch {epoch + 1:03d} | loss {loss_sum / batches:.4f} "
                f"val AUROC {val_auc:.4f} AUPRC {val_ap:.4f} | "
                f"@0.5 ACC {val_acc:.4f} Precision {val_precision:.4f} "
                f"Recall {val_recall:.4f} Specificity {val_specificity:.4f} "
                f"F1 {val_f1:.4f} | "
                f"patience {wait}/{cfg.patience}"
            )
        if wait >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("训练过程中未得到有效模型")
    base_model.load_state_dict(best_state)
    model = base_model

    print(
        f"最佳epoch: {best_epoch}, val AUPRC: {best_score:.4f}, "
        f"划分种子: {cfg.split_seed}, 训练种子: {cfg.seed}"
    )

    test_y, test_prob = predict(model, test_loader)
    test_auc = binary_auroc(test_y, test_prob)
    test_ap = average_precision(test_y, test_prob)
    threshold = 0.5
    test_metrics = threshold_metrics(test_y, test_prob, threshold)

    print("\nFinal held-out test metrics (fixed threshold = 0.5)")
    print(f"ACC        : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision  : {test_metrics['precision'] * 100:.2f}%")
    print(f"Recall     : {test_metrics['recall'] * 100:.2f}%")
    print(f"Specificity: {test_metrics['specificity'] * 100:.2f}%")
    print(f"F1         : {test_metrics['f1'] * 100:.2f}%")
    print(f"AUROC      : {test_auc * 100:.2f}%")
    print(f"AUPRC      : {test_ap * 100:.2f}%")
    print("Confusion matrix [True-IED, Non-IED]:")
    print(test_metrics["confusion_matrix"])
    return {
        "seed": cfg.seed,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "specificity": test_metrics["specificity"],
        "f1": test_metrics["f1"],
        "auroc": test_auc,
        "auprc": test_ap,
        "threshold": threshold,
        "confusion_matrix": test_metrics["confusion_matrix"],
    }


def run_multiple_seeds(cfg: Config, seeds=RANDOM_SEEDS):
    """使用不同随机种子重复完整实验，并报告各指标的均值和样本标准差。"""
    # if len(seeds) != 10:
    #     raise ValueError(f"当前要求运行10个随机种子，实际收到{len(seeds)}个")
    # if len(set(seeds)) != len(seeds):
    #     raise ValueError("随机种子中存在重复值")

    results = []
    for run_index, seed in enumerate(seeds, start=1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = int(seed)
        print(f"\n开始第 {run_index}/{len(seeds)} 次实验")
        results.append(train(run_cfg))

        # 及时释放本轮不再使用的CUDA缓存，降低连续训练10次时的显存压力。
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auroc",
        "auprc",
    )

    print("\n" + "=" * 20 + " 10-seed results " + "=" * 20)
    print(f"Seeds: {list(seeds)}")
    for result in results:
        print(
            f"Seed {result['seed']:>3d} | "
            + " | ".join(
                f"{name.upper()} {result[name] * 100:.2f}%"
                for name in metric_names
            )
            + f" | Threshold {result['threshold']:.3f}"
        )

    summary = {}
    print("\nMean ± sample standard deviation")
    for name in metric_names:
        values = np.asarray([result[name] for result in results], dtype=np.float64)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1))
        summary[name] = {"mean": mean, "std": std}
        print(f"{name.upper():<11}: {mean * 100:.2f} ± {std * 100:.2f}%")

    thresholds = np.asarray(
        [result["threshold"] for result in results], dtype=np.float64
    )
    summary["threshold"] = {
        "mean": float(np.mean(thresholds)),
        "std": float(np.std(thresholds, ddof=1)),
    }
    print(
        f"{'THRESHOLD':<11}: {summary['threshold']['mean']:.3f} ± "
        f"{summary['threshold']['std']:.3f}"
    )

    return results, summary


if __name__ == "__main__":
    run_multiple_seeds(Config())


# ==================== 10-seed results ====================
# Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# Seed  42 | ACCURACY 93.79% | PRECISION 96.02% | RECALL 90.89% | SPECIFICITY 96.49% | F1 93.38% | AUROC 98.53% | AUPRC 98.00% | Threshold 0.500
# Seed  43 | ACCURACY 93.09% | PRECISION 94.00% | RECALL 91.51% | SPECIFICITY 94.56% | F1 92.74% | AUROC 97.90% | AUPRC 97.23% | Threshold 0.500
# Seed  44 | ACCURACY 95.00% | PRECISION 94.40% | RECALL 95.29% | SPECIFICITY 94.73% | F1 94.84% | AUROC 98.95% | AUPRC 98.76% | Threshold 0.500
# Seed  45 | ACCURACY 92.94% | PRECISION 96.19% | RECALL 88.87% | SPECIFICITY 96.72% | F1 92.39% | AUROC 98.24% | AUPRC 97.73% | Threshold 0.500
# Seed  46 | ACCURACY 93.15% | PRECISION 95.84% | RECALL 89.69% | SPECIFICITY 96.37% | F1 92.66% | AUROC 98.17% | AUPRC 97.11% | Threshold 0.500
# Seed  47 | ACCURACY 94.79% | PRECISION 94.99% | RECALL 94.15% | SPECIFICITY 95.38% | F1 94.57% | AUROC 98.46% | AUPRC 97.84% | Threshold 0.500
# Seed  48 | ACCURACY 93.70% | PRECISION 95.89% | RECALL 90.82% | SPECIFICITY 96.37% | F1 93.29% | AUROC 98.38% | AUPRC 97.91% | Threshold 0.500
# Seed  49 | ACCURACY 93.48% | PRECISION 95.03% | RECALL 91.26% | SPECIFICITY 95.55% | F1 93.11% | AUROC 98.17% | AUPRC 97.77% | Threshold 0.500
# Seed  50 | ACCURACY 93.18% | PRECISION 95.96% | RECALL 89.63% | SPECIFICITY 96.49% | F1 92.69% | AUROC 98.49% | AUPRC 98.27% | Threshold 0.500
# Seed  51 | ACCURACY 94.27% | PRECISION 92.74% | RECALL 95.60% | SPECIFICITY 93.04% | F1 94.15% | AUROC 98.48% | AUPRC 98.20% | Threshold 0.500
#
# Mean ± sample standard deviation
# ACCURACY   : 93.74 ± 0.73%
# PRECISION  : 95.10 ± 1.12%
# RECALL     : 91.77 ± 2.40%
# SPECIFICITY: 95.57 ± 1.18%
# F1         : 93.38 ± 0.86%
# AUROC      : 98.38 ± 0.28%
# AUPRC      : 97.88 ± 0.48%