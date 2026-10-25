import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from .. import config

BINARY_MAP = {0: 0, 1: 0, 2: 1, 3: 1}


def _predict_batch(model, pixel_values, is_unet):
    if is_unet:
        logits = model(pixel_values)
    else:
        logits = model(pixel_values=pixel_values).logits
    return logits


def evaluate_segmentation(model, val_ds, device, is_unet=False, max_items=None, batch_size=4):
    model.eval()
    n = len(val_ds) if max_items is None else min(max_items, len(val_ds))
    all_preds, all_labels = [], []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idxs = range(start, min(start + batch_size, n))
            batch = [val_ds[i] for i in idxs]
            pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)

            logits = _predict_batch(model, pixel_values, is_unet)
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = upsampled.argmax(dim=1)

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    valid = y_true != 255
    y_pred, y_true = y_pred[valid], y_true[valid]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_LABELS)))
    ious = []
    for c in range(config.NUM_LABELS):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        ious.append(iou)

    return {"confusion_matrix": cm, "per_class_iou": dict(zip(config.CLASS_NAMES, ious)), "mean_iou": float(np.nanmean(ious))}


def evaluate_binary_passability(model, val_ds, device, is_unet=False, max_items=None, batch_size=4):
    model.eval()
    n = len(val_ds) if max_items is None else min(max_items, len(val_ds))
    all_preds, all_labels = [], []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idxs = range(start, min(start + batch_size, n))
            batch = [val_ds[i] for i in idxs]
            pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(device)
            labels = torch.stack([b["labels"] for b in batch]).to(device)

            logits = _predict_batch(model, pixel_values, is_unet)
            upsampled = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            preds = upsampled.argmax(dim=1)

            all_preds.append(preds.cpu().numpy().ravel())
            all_labels.append(labels.cpu().numpy().ravel())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    valid = y_true != 255
    y_pred, y_true = y_pred[valid], y_true[valid]

    y_pred_bin = np.vectorize(BINARY_MAP.get)(y_pred)
    y_true_bin = np.vectorize(BINARY_MAP.get)(y_true)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"recall": recall, "precision": precision}
