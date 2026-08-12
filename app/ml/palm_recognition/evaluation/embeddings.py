"""
Biometric Evaluation Pipeline untuk PalmNet-Lite.

Phase 3: Validation Calibration (threshold dari val set)
Phase 4: Final Hold-out Evaluation (test identities 401-600)

Protocol:
    Test set: Palm 401-600 (UNSEEN during training)
    Session 1 -> enrollment template
    Session 2 -> query

Session diambil dari kolom 'session' di CSV, bukan sort-50/50.

Metrics:
    Rank-1 Accuracy, EER, ROC-AUC, TAR@FAR 0.1%, TAR@FAR 0.01%
    mean genuine similarity, mean impostor similarity, cosine gap
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings_from_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized embeddings dari DataLoader.

    Returns:
        (embeddings [N, 128], labels [N,])
    """
    model.eval()
    all_embs = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        embs = model(images)
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
        all_labels.extend(labels.tolist())

    return np.concatenate(all_embs, axis=0), np.array(all_labels)


@torch.no_grad()
def extract_embeddings_from_paths(
    model: nn.Module,
    image_paths: list[str],
    transform,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract embeddings dari list of image paths, batched.

    Returns:
        [N, 128] L2-normalized embeddings
    """
    model.eval()
    all_embs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img_np = np.array(img)
            try:
                t = transform(image=img_np)["image"]
            except (TypeError, KeyError):
                t = transform(img)
            batch_tensors.append(t)

        batch = torch.stack(batch_tensors).to(device)
        embs = model(batch)
        embs = F.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def build_enrollment_templates(
    embeddings: np.ndarray,
    class_indices: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Bangun enrollment template per identity.

    Protocol:
        1. L2-normalize setiap embedding
        2. Average embeddings per identity
        3. L2-normalize template kembali

    Returns:
        [num_classes, 128] unit vectors
    """
    templates = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)

    for emb, cls_idx in zip(embeddings, class_indices):
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            templates[cls_idx] += emb / norm
        counts[cls_idx] += 1

    # Average dan normalize template
    for c in range(num_classes):
        if counts[c] > 0:
            templates[c] /= counts[c]
            n = np.linalg.norm(templates[c])
            if n > 1e-8:
                templates[c] /= n

    return templates


def evaluate_biometric_session_aware(
    model: nn.Module,
    test_csv: str | Path,
    transform,
    device: torch.device,
    far_targets: Optional[list[float]] = None,
) -> dict:
    """Evaluasi biometric dengan session-aware protocol.

    Menggunakan kolom 'session' di test.csv secara eksplisit:
        Session 1 -> enrollment
        Session 2 -> query

    TIDAK melakukan sort-path 50/50.

    Args:
        model:      backbone (akan dipanggil F.normalize setelah forward)
        test_csv:   path ke test.csv dengan kolom path,label,palm_id,session
        transform:  val_transform (deterministic)
        device:     torch device
        far_targets: list FAR targets untuk TAR@FAR, default [0.001, 0.0001]

    Returns:
        dict dengan semua metrics (tanpa array besar, arrays di _fpr dll)
    """
    if far_targets is None:
        far_targets = [0.001, 0.0001]

    from sklearn.metrics import roc_curve, auc as sklearn_auc

    df = pd.read_csv(test_csv)
    required = {"path", "label", "session"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"test.csv missing kolom: {missing_cols}")

    # Pisahkan enrollment (session 1) dan query (session 2)
    df_enroll = df[df["session"] == 1].copy()
    df_query  = df[df["session"] == 2].copy()

    if len(df_enroll) == 0 or len(df_query) == 0:
        raise ValueError("test.csv harus punya session 1 (enroll) dan session 2 (query)")

    # Build class index map dari label kolom (string untuk test set)
    all_labels_unique = sorted(df["label"].astype(str).unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels_unique)}
    num_classes = len(label_to_idx)

    print(f"Test set: {num_classes} identities, {len(df_enroll)} enroll, {len(df_query)} query")

    # Extract embeddings enrollment
    enroll_paths = df_enroll["path"].tolist()
    enroll_labels = [label_to_idx[str(l)] for l in df_enroll["label"]]
    print("Extracting enrollment embeddings...")
    enroll_embs = extract_embeddings_from_paths(model, enroll_paths, transform, device)

    # Extract embeddings query
    query_paths = df_query["path"].tolist()
    query_labels = [label_to_idx[str(l)] for l in df_query["label"]]
    print("Extracting query embeddings...")
    query_embs = extract_embeddings_from_paths(model, query_paths, transform, device)

    enroll_labels_np = np.array(enroll_labels)
    query_labels_np  = np.array(query_labels)

    # Build enrollment templates
    templates = build_enrollment_templates(enroll_embs, enroll_labels_np, num_classes)

    # Score matrix: query vs templates [num_query, num_classes]
    score_matrix = query_embs @ templates.T

    # Genuine dan impostor scores
    genuine_scores = []
    impostor_scores = []
    for q_idx, true_cls in enumerate(query_labels_np):
        for c in range(num_classes):
            s = float(score_matrix[q_idx, c])
            if c == true_cls:
                genuine_scores.append(s)
            else:
                impostor_scores.append(s)

    genuine_scores = np.array(genuine_scores, dtype=np.float32)
    impostor_scores = np.array(impostor_scores, dtype=np.float32)

    # Rank-1 accuracy
    pred_class = score_matrix.argmax(axis=1)
    rank1_acc = float((pred_class == query_labels_np).mean())

    # ROC + EER
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = float(sklearn_auc(fpr, tpr))

    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    tar_at_far = {}
    for far_target in far_targets:
        idx = int(np.argmin(np.abs(fpr - far_target)))
        tar_at_far[f"tar_at_far_{far_target}"] = float(tpr[idx])

    results = {
        "num_classes": num_classes,
        "num_enroll_images": len(df_enroll),
        "num_query_images": len(df_query),
        "num_genuine_pairs": int(len(genuine_scores)),
        "num_impostor_pairs": int(len(impostor_scores)),
        "rank1_accuracy": rank1_acc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "roc_auc": roc_auc,
        "mean_genuine_score": float(genuine_scores.mean()),
        "std_genuine_score": float(genuine_scores.std()),
        "mean_impostor_score": float(impostor_scores.mean()),
        "std_impostor_score": float(impostor_scores.std()),
        "cosine_gap": float(genuine_scores.mean() - impostor_scores.mean()),
        **tar_at_far,
        # Arrays untuk plotting
        "_fpr": fpr,
        "_tpr": tpr,
        "_thresholds": thresholds,
        "_genuine_scores": genuine_scores,
        "_impostor_scores": impostor_scores,
    }
    return results


def calibrate_threshold_from_val(
    model: nn.Module,
    val_csv: str | Path,
    transform,
    device: torch.device,
    model_id: str = "palmnet-lite-scratch",
    version: str = "1.0.0",
) -> dict:
    """Kalibrasi threshold dari validation data (BUKAN test set).

    Threshold ditentukan di EER point.

    Returns:
        threshold.json content dict
    """
    df = pd.read_csv(val_csv)
    all_labels_unique = sorted(df["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels_unique)}
    num_classes = len(label_to_idx)

    paths = df["path"].tolist()
    labels = [label_to_idx[l] for l in df["label"]]

    print(f"Calibration: extracting embeddings dari {len(paths)} val images...")
    embs = extract_embeddings_from_paths(model, paths, transform, device)

    labels_np = np.array(labels)
    genuine_scores = []
    impostor_scores = []

    rng = np.random.default_rng(42)
    label_to_emb_idx: dict = defaultdict(list)
    for i, lbl in enumerate(labels_np):
        label_to_emb_idx[int(lbl)].append(i)

    class_list = list(label_to_emb_idx.keys())
    for cls_idx, idxs in label_to_emb_idx.items():
        idxs = np.array(idxs)
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                s = float(np.dot(embs[idxs[i]], embs[idxs[j]]))
                genuine_scores.append(s)

    # Impostor: sample
    n_imp = min(len(genuine_scores) * 3, 100000)
    for _ in range(n_imp * 2):
        if len(impostor_scores) >= n_imp:
            break
        c1, c2 = rng.choice(len(class_list), 2, replace=False)
        i1 = rng.choice(label_to_emb_idx[class_list[c1]])
        i2 = rng.choice(label_to_emb_idx[class_list[c2]])
        s = float(np.dot(embs[i1], embs[i2]))
        impostor_scores.append(s)

    from sklearn.metrics import roc_curve
    all_scores = np.concatenate([
        np.array(genuine_scores), np.array(impostor_scores)
    ])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))
    ])

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return {
        "model_id": model_id,
        "version": version,
        "metric": "cosine_similarity",
        "threshold": eer_threshold,
        "calibration_split": "validation",
        "selection": "eer",
        "eer": eer,
        "mean_genuine": float(np.mean(genuine_scores)),
        "mean_impostor": float(np.mean(impostor_scores)),
        "n_genuine_pairs": len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
    }
