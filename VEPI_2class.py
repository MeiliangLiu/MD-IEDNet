import copy
import glob
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# from model import CausalIEDGraphNet
from model import CausalIEDGraphNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEEDS = tuple(range(42, 52))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # 必改：数据目录。
    data_dir: str = r"D:\Epilepsy-2class"

    # 文件名示例：DA00100A_0_2000_500_0.npy，其中第一个下划线前为患者编号。
    # 同一患者的全部片段只会出现在train、val、test中的一个集合。
    group_regex: Optional[str] = r"^([^_]+)_"

    target_len: int = 2000
    channels: int = 19
    time_bins: int = 16
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15

    # 通常只需要调下面两个参数，其他结构参数建议固定。
    lr: float = 5e-3
    dropout: float = 0.2
    # lr: float = 1e-3
    # dropout: float = 0.1

    weight_decay: float = 1e-4
    node_dim: int = 48
    graph_hidden: int = 128
    rank_dim: int = 12
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    # 固定数据划分，使10次实验只改变模型训练随机性。
    split_seed: int = 42
    seed: int = 42
    compile_model: bool = False
    checkpoint_path: str = "best_multiscale_simplified_causal_ied_graphnet.pt"


class InMemoryEEGDataset(Dataset):
    classes = ("True-IED", "Non-IED")

    def __init__(
        self,
        data_dir: str,
        target_len: int = 2000,
        channels: int = 19,
        group_regex: Optional[str] = None,
    ):
        self.labels = []
        self.groups = []
        arrays = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))
            if not files:
                raise FileNotFoundError(f"类别目录为空或不存在: {class_dir}")

            for path in files:
                x = np.load(path, allow_pickle=False)
                if x.ndim != 2:
                    raise ValueError(f"{path} 应为二维数组，实际形状为 {x.shape}")
                if x.shape[0] != channels and x.shape[1] == channels:
                    x = x.T
                if x.shape[0] < channels:
                    raise ValueError(f"{path} 只有 {x.shape[0]} 个通道，少于 {channels}")

                x = np.asarray(x[:channels], dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError(f"{path} 含 NaN 或 Inf")

                length = x.shape[1]
                if length > target_len:
                    start = (length - target_len) // 2
                    x = x[:, start : start + target_len]

                # 每段、每通道标准化；padding后的0对应标准化均值。
                x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(
                    x.std(axis=1, keepdims=True), 1e-6
                )
                if length < target_len:
                    pad = target_len - length
                    left = pad // 2
                    x = np.pad(x, ((0, 0), (left, pad - left)), mode="constant")

                arrays.append(torch.from_numpy(np.ascontiguousarray(x)))
                self.labels.append(label)

                basename = os.path.basename(path)
                if group_regex is None:
                    self.groups.append(basename)
                else:
                    match = re.search(group_regex, basename)
                    if match is None:
                        raise ValueError(f"文件名无法匹配 group_regex: {basename}")
                    self.groups.append(match.group(1) if match.groups() else match.group(0))

        self.data = torch.stack(arrays)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.groups = np.asarray(self.groups)
        print(f"已加载 {len(self.labels)} 个样本，形状: {tuple(self.data.shape)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])


def stratified_indices(labels, val_ratio, test_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_ratio))
        n_val = max(1, round(len(idx) * val_ratio))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return list(train_idx), list(val_idx), list(test_idx)


def group_stratified_indices(labels, groups, val_ratio, test_ratio, seed):
    """无额外依赖的患者级贪心分层划分。"""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    if len(unique_groups) < 3:
        raise ValueError("患者组少于3组，无法划分train/val/test")

    group_counts = {
        group: np.bincount(labels[groups == group], minlength=2).astype(np.float64)
        for group in unique_groups
    }
    total = np.bincount(labels, minlength=2).astype(np.float64)
    target = np.stack(
        [total * (1.0 - val_ratio - test_ratio), total * val_ratio, total * test_ratio]
    )
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
        for split in range(3):
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

    indices = [[], [], []]
    for idx, group in enumerate(groups):
        indices[assignment[group]].append(idx)
    if any(len(split) == 0 for split in indices):
        raise RuntimeError("患者组分布不适合当前比例，某个数据集为空")
    return indices[0], indices[1], indices[2]


def make_splits(dataset: InMemoryEEGDataset, cfg: Config):
    if cfg.group_regex is None:
        print("警告：当前为样本级划分，仅可用于代码试运行；正式结果请设置 group_regex。")
        return stratified_indices(
            dataset.labels, cfg.val_ratio, cfg.test_ratio, cfg.split_seed
        )
    splits = group_stratified_indices(
        dataset.labels,
        dataset.groups,
        cfg.val_ratio,
        cfg.test_ratio,
        cfg.split_seed,
    )
    group_sets = [set(dataset.groups[idx]) for idx in splits]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("患者级划分失败：不同集合间出现重复患者")

    print(f"共识别到 {len(np.unique(dataset.groups))} 名患者")
    for split_name, indices, patient_ids in zip(
        ("train", "val", "test"), splits, group_sets
    ):
        class_counts = np.bincount(dataset.labels[indices], minlength=2)
        print(
            f"{split_name}: 患者={len(patient_ids)}, 样本={len(indices)}, "
            f"[True-IED, Non-IED]={class_counts.tolist()}"
        )
    print("患者级划分检查通过：train、val、test之间无重复患者")
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


def make_loaders(dataset, split_indices, cfg):
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    train_idx, val_idx, test_idx = split_indices
    return (
        DataLoader(Subset(dataset, train_idx), shuffle=True, drop_last=False, **common),
        DataLoader(Subset(dataset, val_idx), shuffle=False, **common),
        DataLoader(Subset(dataset, test_idx), shuffle=False, **common),
    )


def train(cfg: Config):
    print(f"\n{'=' * 24} Seed {cfg.seed} {'=' * 24}")
    set_seed(cfg.seed)
    dataset = InMemoryEEGDataset(
        cfg.data_dir, cfg.target_len, cfg.channels, cfg.group_regex
    )
    split_indices = make_splits(dataset, cfg)
    train_loader, val_loader, test_loader = make_loaders(dataset, split_indices, cfg)
    train_counts = np.bincount(dataset.labels[split_indices[0]], minlength=2)
    print(
        f"train/val/test: {len(split_indices[0])}/{len(split_indices[1])}/{len(split_indices[2])}"
    )
    print(f"训练集类别计数 [True-IED, Non-IED]: {train_counts.tolist()}")

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
        (
            val_acc,
            val_precision,
            val_recall,
            val_specificity,
            val_f1,
        ) = metrics_from_cm(confusion_matrix(val_y, fixed_pred))

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

    threshold = 0.5
    print(
        f"最佳epoch: {best_epoch}, val AUPRC: {best_score:.4f}, "
        f"划分种子: {cfg.split_seed}, 训练种子: {cfg.seed}, "
        f"固定IED阈值: {threshold:.3f}"
    )

    test_y, test_prob = predict(model, test_loader)
    test_pred = np.where(test_prob >= threshold, 0, 1)
    test_cm = confusion_matrix(test_y, test_pred)
    (
        test_acc,
        test_precision,
        test_recall,
        test_specificity,
        test_f1,
    ) = metrics_from_cm(test_cm)
    test_auc = binary_auroc(test_y, test_prob)
    test_ap = average_precision(test_y, test_prob)

    print("\nFinal held-out test metrics (fixed threshold = 0.5)")
    print(f"ACC       : {test_acc * 100:.2f}%")
    print(f"Precision : {test_precision * 100:.2f}%")
    print(f"Recall    : {test_recall * 100:.2f}%")
    print(f"Specificity: {test_specificity * 100:.2f}%")
    print(f"F1        : {test_f1 * 100:.2f}%")
    print(f"AUROC     : {test_auc * 100:.2f}%")
    print(f"AUPRC     : {test_ap * 100:.2f}%")
    print("Confusion matrix [True-IED, Non-IED]:")
    print(test_cm)
    return {
        "seed": cfg.seed,
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "specificity": test_specificity,
        "f1": test_f1,
        "auroc": test_auc,
        "auprc": test_ap,
        "threshold": threshold,
        "confusion_matrix": test_cm,
    }


def run_multiple_seeds(cfg: Config, seeds=RANDOM_SEEDS):
    """使用10个训练随机种子重复实验，并报告均值和样本标准差。"""
    if len(seeds) != 10:
        raise ValueError(f"当前要求运行10个随机种子，实际收到{len(seeds)}个")
    if len(set(seeds)) != len(seeds):
        raise ValueError("随机种子中存在重复值")

    results = []
    for run_index, seed in enumerate(seeds, start=1):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.seed = int(seed)
        print(f"\n开始第 {run_index}/{len(seeds)} 次实验")
        results.append(train(run_cfg))

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
        values = np.asarray(
            [result[name] for result in results], dtype=np.float64
        )
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



# #==================== 10-seed results ====================
# Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
# Seed  42 | ACCURACY 93.28% | PRECISION 65.58% | RECALL 58.14% | SPECIFICITY 96.88% | F1 61.64% | AUROC 90.35% | AUPRC 67.35% | Threshold 0.500
# Seed  43 | ACCURACY 93.93% | PRECISION 75.45% | RECALL 51.34% | SPECIFICITY 98.29% | F1 61.10% | AUROC 91.04% | AUPRC 68.07% | Threshold 0.500
# Seed  44 | ACCURACY 93.05% | PRECISION 63.20% | RECALL 60.21% | SPECIFICITY 96.41% | F1 61.67% | AUROC 90.77% | AUPRC 66.61% | Threshold 0.500
# Seed  45 | ACCURACY 93.78% | PRECISION 75.32% | RECALL 49.07% | SPECIFICITY 98.35% | F1 59.43% | AUROC 91.80% | AUPRC 67.84% | Threshold 0.500
# Seed  46 | ACCURACY 93.51% | PRECISION 81.74% | RECALL 38.76% | SPECIFICITY 99.11% | F1 52.59% | AUROC 89.23% | AUPRC 64.10% | Threshold 0.500
# Seed  47 | ACCURACY 93.80% | PRECISION 69.88% | RECALL 58.35% | SPECIFICITY 97.43% | F1 63.60% | AUROC 90.84% | AUPRC 67.92% | Threshold 0.500
# Seed  48 | ACCURACY 92.88% | PRECISION 60.89% | RECALL 65.15% | SPECIFICITY 95.72% | F1 62.95% | AUROC 89.74% | AUPRC 68.03% | Threshold 0.500
# Seed  49 | ACCURACY 93.59% | PRECISION 67.12% | RECALL 60.62% | SPECIFICITY 96.96% | F1 63.71% | AUROC 89.12% | AUPRC 67.09% | Threshold 0.500
# Seed  50 | ACCURACY 93.68% | PRECISION 68.07% | RECALL 60.21% | SPECIFICITY 97.11% | F1 63.89% | AUROC 89.31% | AUPRC 67.38% | Threshold 0.500
# Seed  51 | ACCURACY 93.53% | PRECISION 65.74% | RECALL 63.30% | SPECIFICITY 96.62% | F1 64.50% | AUROC 90.11% | AUPRC 67.97% | Threshold 0.500
#
# Mean ± sample standard deviation
# ACCURACY   : 93.50 ± 0.34%
# PRECISION  : 69.30 ± 6.41%
# RECALL     : 56.52 ± 7.94%
# SPECIFICITY: 97.29 ± 1.03%
# F1         : 61.51 ± 3.50%
# AUROC      : 90.23 ± 0.89%
# AUPRC      : 67.24 ± 1.20%