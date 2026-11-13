"""VEPI 五类 IED 空间分类：患者级固定划分，10 个训练随机种子。

把本文件和 model_5ied.py 放在同一目录，在 PyCharm 运行本文件。
仅读取 D:\\Epilepsy 下五个 *-IED 子目录；不读取 non-IED。
需要 numpy、torch。训练结果和患者划分保存至 RESULT_DIR。
"""

import copy
import csv
import glob
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from model_5ied import CausalIEDGraphNet


CLASS_NAMES = (
    "Generalized-IED",
    "Frontal-IED",
    "Temporal-IED",
    "Centro-Parietal-IED",
    "Occipital-IED",
)
N_CLASSES = len(CLASS_NAMES)
SEEDS = tuple(range(42, 52))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    data_dir: str = r"D:\Epilepsy"
    result_dir: str = r"D:\IED_benchmark\Ours\5class"
    # 文件名示例 DA00100A_0_2000_500_1.npy：第一个下划线前为患者 ID。
    group_regex: str = r"^([^_]+)_"
    target_len: int = 2000
    channels: int = 19
    time_bins: int = 16
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15
    lr: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    node_dim: int = 48
    graph_hidden: int = 128
    rank_dim: int = 12
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    split_seed: int = 42
    split_attempts: int = 5000
    compile_model: bool = False
    use_class_weights: bool = False  # True 时按训练集样本数的平方根倒数加权
    save_checkpoints: bool = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IEDDataset(Dataset):
    def __init__(self, cfg):
        arrays, labels, groups, paths = [], [], [], []
        for label, name in enumerate(CLASS_NAMES):
            folder = os.path.join(cfg.data_dir, name)
            files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            if not files:
                raise FileNotFoundError(f"类别目录不存在或没有 NPY 文件: {folder}")
            for path in files:
                basename = os.path.basename(path)
                match = re.search(cfg.group_regex, basename)
                if match is None:
                    raise ValueError(f"无法从文件名提取患者 ID: {basename}")
                x = np.load(path, allow_pickle=False)
                if x.ndim != 2:
                    raise ValueError(f"{path}: 需要二维数组，实际 {x.shape}")
                # 接受 (19 或 29, 时间) 及其转置；29 路仅保留前 19 路。
                if x.shape[0] in (19, 29):
                    pass
                elif x.shape[1] in (19, 29):
                    x = x.T
                else:
                    raise ValueError(f"{path}: 无法识别 19/29 通道，实际 {x.shape}")
                if x.shape[0] < cfg.channels:
                    raise ValueError(f"{path}: 通道数不足 {cfg.channels}")
                x = np.asarray(x[: cfg.channels], dtype=np.float32)
                if not np.isfinite(x).all() or x.shape[1] < 2:
                    raise ValueError(f"{path}: 信号含 NaN/Inf 或时间点不足")

                length = x.shape[1]
                if length > cfg.target_len:
                    left = (length - cfg.target_len) // 2
                    x = x[:, left : left + cfg.target_len]
                # 与原二分类脚本一致：每段每通道标准化后，居中填充零。
                x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(
                    x.std(axis=1, keepdims=True), 1e-6
                )
                if x.shape[1] < cfg.target_len:
                    padding = cfg.target_len - x.shape[1]
                    x = np.pad(x, ((0, 0), (padding // 2, padding - padding // 2)))
                arrays.append(torch.from_numpy(np.ascontiguousarray(x)))
                labels.append(label)
                groups.append(match.group(1) if match.groups() else match.group(0))
                paths.append(path)

        self.data = torch.stack(arrays)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.asarray(groups)
        self.paths = paths
        print(f"读取 {len(self.labels)} 个 IED 样本，张量形状 {tuple(self.data.shape)}")
        print("类别顺序及数量:", dict(zip(CLASS_NAMES, np.bincount(self.labels, minlength=N_CLASSES).tolist())))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])


def patient_split(ds, cfg):
    """固定患者级划分；从多个随机候选中选类别分布最接近目标的一组。"""
    patients, inv = np.unique(ds.groups, return_inverse=True)
    n_patients = len(patients)
    if n_patients < 3:
        raise ValueError("至少需要 3 名患者才能划分 train/val/test")
    counts = np.zeros((n_patients, N_CLASSES), dtype=np.int64)
    np.add.at(counts, (inv, ds.labels), 1)
    patient_support = (counts > 0).sum(axis=0)
    if (patient_support < 3).any():
        missing = [CLASS_NAMES[i] for i in np.where(patient_support < 3)[0]]
        raise ValueError(f"这些类别出现于不足 3 名患者，无法让三个集合均包含它们: {missing}")
    n_test = max(1, round(n_patients * cfg.test_ratio))
    n_val = max(1, round(n_patients * cfg.val_ratio))
    if n_test + n_val >= n_patients:
        raise ValueError("划分比例使训练集没有患者")

    total = counts.sum(axis=0)
    target = np.stack([
        total * (1 - cfg.val_ratio - cfg.test_ratio),
        total * cfg.val_ratio,
        total * cfg.test_ratio,
    ])
    rng = np.random.default_rng(cfg.split_seed)
    best, best_score = None, float("inf")
    for _ in range(cfg.split_attempts):
        order = rng.permutation(n_patients)
        g_test = order[:n_test]
        g_val = order[n_test : n_test + n_val]
        g_train = order[n_test + n_val :]
        g_indices = (g_train, g_val, g_test)
        actual = np.stack([counts[ids].sum(axis=0) for ids in g_indices])
        if (actual == 0).any():
            continue
        class_cost = np.square((actual - target) / np.maximum(target, 1)).mean()
        size_cost = np.square(
            (actual.sum(axis=1) - target.sum(axis=1))
            / np.maximum(target.sum(axis=1), 1)
        ).mean()
        score = class_cost + 0.25 * size_cost
        if score < best_score:
            best_score, best = score, g_indices
    if best is None:
        raise RuntimeError("未找到每个集合都有五类 IED 的患者级划分；请检查患者和标签分布")

    splits = tuple(np.flatnonzero(np.isin(inv, group_ids)).tolist() for group_ids in best)
    for name, indices in zip(("train", "val", "test"), splits):
        n = np.bincount(ds.labels[indices], minlength=N_CLASSES)
        print(f"{name}: 患者 {len(np.unique(ds.groups[indices]))}, 样本 {len(indices)}, 五类计数 {n.tolist()}")
    group_sets = [set(ds.groups[idx]) for idx in splits]
    assert not any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3))
    return splits


def confusion(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def binary_auc_and_ap(y_binary, scores):
    y_binary = np.asarray(y_binary, dtype=np.int64)
    n_pos, n_neg = int(y_binary.sum()), int(len(y_binary) - y_binary.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_y, sorted_s = y_binary[order], np.asarray(scores)[order]
    last = np.r_[np.flatnonzero(np.diff(sorted_s)), len(sorted_s) - 1]
    tp = np.cumsum(sorted_y)[last]
    fp = (last + 1) - tp
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / n_neg]
    auc = float(np.trapz(tpr, fpr))
    recall = tp / n_pos
    precision = tp / (last + 1)
    ap = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
    return auc, ap


def evaluate(y, prob):
    pred = prob.argmax(axis=1)
    cm = confusion(y, pred)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    tn = cm.sum() - (support + predicted - tp)
    specificity = tn / np.maximum(tn + predicted - tp, 1)
    auc_ap = [binary_auc_and_ap(y == i, prob[:, i]) for i in range(N_CLASSES)]
    aucs, aps = np.asarray(auc_ap).T
    metrics = {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_specificity": float(specificity.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auroc_ovr": float(aucs.mean()),
        "macro_auprc_ovr": float(aps.mean()),
    }
    per_class = {
        name: dict(support=int(support[i]), precision=float(precision[i]),
                   recall=float(recall[i]), specificity=float(specificity[i]),
                   f1=float(f1[i]), auroc_ovr=float(aucs[i]),
                   auprc_ovr=float(aps[i]))
        for i, name in enumerate(CLASS_NAMES)
    }
    return metrics, per_class, cm


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        logits, _ = model(x.to(DEVICE, non_blocking=True))
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def train_one_seed(cfg, ds, splits, seed):
    print(f"\n{'=' * 20} Seed {seed} {'=' * 20}")
    set_seed(seed)
    options = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=DEVICE.type == "cuda")
    train_loader = DataLoader(Subset(ds, splits[0]), shuffle=True, **options)
    val_loader = DataLoader(Subset(ds, splits[1]), shuffle=False, **options)
    test_loader = DataLoader(Subset(ds, splits[2]), shuffle=False, **options)
    base_model = CausalIEDGraphNet(cfg).to(DEVICE)
    model = torch.compile(base_model) if cfg.compile_model and hasattr(torch, "compile") else base_model
    if cfg.use_class_weights:
        train_counts = np.bincount(ds.labels[splits[0]], minlength=N_CLASSES)
        weights = 1.0 / np.sqrt(train_counts.astype(float))
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.10, div_factor=10.0, final_div_factor=100.0,
    )
    best_f1, best_epoch, best_state, wait = -1.0, -1, None, 0
    for epoch in range(1, cfg.epochs + 1):
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
            loss_sum += loss.item()
        val_y, val_prob = predict(model, val_loader)
        val_metrics, _, _ = evaluate(val_y, val_prob)
        if val_metrics["macro_f1"] > best_f1 + 1e-12:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(base_model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0 or wait >= cfg.patience:
            print(f"Epoch {epoch:03d} | loss {loss_sum / len(train_loader):.4f} | "
                  f"val ACC {val_metrics['accuracy']:.4f} | "
                  f"val macro-F1 {val_metrics['macro_f1']:.4f} | "
                  f"patience {wait}/{cfg.patience}")
        if wait >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("训练未选出有效模型")
    base_model.load_state_dict(best_state)
    y, prob = predict(base_model, test_loader)
    metrics, per_class, cm = evaluate(y, prob)
    print(f"Best epoch {best_epoch}, val macro-F1 {best_f1:.4f}")
    print("Test:", " | ".join(f"{k}={v * 100:.2f}%" for k, v in metrics.items()))
    print("混淆矩阵（行=真实类别，列=预测类别；顺序见 CLASS_NAMES）:\n", cm)
    for name in CLASS_NAMES:
        cls = per_class[name]
        print(f"  {name:<21} n={cls['support']:>4} "
              f"P={cls['precision']:.4f} R={cls['recall']:.4f} F1={cls['f1']:.4f}")
    return {"seed": seed, "best_epoch": best_epoch, "metrics": metrics,
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = Config()
    if len(SEEDS) != 10 or len(set(SEEDS)) != 10:
        raise ValueError("请指定 10 个不重复训练随机种子")
    os.makedirs(cfg.result_dir, exist_ok=True)
    ds = IEDDataset(cfg)
    splits = patient_split(ds, cfg)
    write_csv(
        os.path.join(cfg.result_dir, "patient_split.csv"),
        ["split", "patient_id", "class_name", "file_path"],
        [{"split": split, "patient_id": ds.groups[idx],
          "class_name": CLASS_NAMES[ds.labels[idx]], "file_path": ds.paths[idx]}
         for split, indices in zip(("train", "val", "test"), splits) for idx in indices],
    )
    results = []
    for seed in SEEDS:
        results.append(train_one_seed(cfg, ds, splits, seed))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    keys = tuple(results[0]["metrics"])
    write_csv(os.path.join(cfg.result_dir, "seed_metrics.csv"),
              ["seed", "best_epoch", *keys],
              [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                **r["metrics"]} for r in results])
    class_rows = []
    for result in results:
        for name, values in result["per_class"].items():
            class_rows.append({"seed": result["seed"], "class_name": name, **values})
        with open(os.path.join(cfg.result_dir, f"confusion_seed{result['seed']}.csv"),
                  "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true \\ predicted", *CLASS_NAMES])
            for name, row in zip(CLASS_NAMES, result["confusion_matrix"]):
                writer.writerow([name, *row])
    write_csv(os.path.join(cfg.result_dir, "per_class_metrics.csv"),
              ["seed", "class_name", "support", "precision", "recall",
               "specificity", "f1", "auroc_ovr", "auprc_ovr"], class_rows)

    print("\n=== 10-seed summary: mean ± sample SD (%) ===")
    summary = {}
    for key in keys:
        values = np.asarray([r["metrics"][key] for r in results], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        print(f"{key:<19} {100 * values.mean():.2f} ± {100 * values.std(ddof=1):.2f}")
    with open(os.path.join(cfg.result_dir, "summary_5ied.json"),
              "w", encoding="utf-8") as handle:
        json.dump({"class_names": CLASS_NAMES, "split_seed": cfg.split_seed,
                   "seeds": SEEDS, "summary": summary, "results": results},
                  handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
