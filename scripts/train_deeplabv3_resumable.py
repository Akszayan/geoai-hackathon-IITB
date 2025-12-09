"""
Resumable DeepLabV3 training script with robust logging, GPU stats, safe optimizer parsing,
backbone-only pretrained weights, resume support, graceful Ctrl+C, and clean validation metrics.

Run:
    python -u scripts/train_deeplabv3_resumable.py --config configs/train_deeplabv3_pb.yaml
"""

import argparse
import time
import math
import os
import sys
import random
import signal
import traceback
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50


# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def show_gpu_stats(device):
    if device.type == "cuda":
        try:
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            safe_print(f"   GPU mem: alloc={alloc:.1f}MB reserved={reserved:.1f}MB")
        except Exception as e:
            safe_print("[WARN] GPU stats unavailable:", e)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def dice_loss(pred_logits, target, eps=1e-6):
    pred = F.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1])\
        .permute(0, 3, 1, 2).float()

    inter = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def iou_per_class_np(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((inter / union).item())
    return ious


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class NpySegDataset(Dataset):
    def __init__(self, list_txt, mean, std, augment=False, debug_max_batches=0):
        self.list_txt = Path(list_txt)
        if not self.list_txt.exists():
            raise FileNotFoundError(f"List file does not exist: {self.list_txt}")

        self.samples = [l.strip() for l in open(self.list_txt) if l.strip()]

        if debug_max_batches > 0:
            self.samples = self.samples[:max(1, debug_max_batches * 2)]

        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        mask_path = img_path.replace("images_npy", "masks_npy")

        img = np.load(img_path).astype("float32") / 255.0   # H,W,3
        mask = np.load(mask_path).astype("int64")           # H,W

        # simple aug
        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, 1).copy()
                mask = np.flip(mask, 1).copy()
            if random.random() < 0.5:
                img = np.flip(img, 0).copy()
                mask = np.flip(mask, 0).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        # normalize
        img = (img - self.mean.reshape(1, 1, 3)) / self.std.reshape(1, 1, 3)
        img = img.transpose(2, 0, 1)  # CHW

        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


# --------------------------------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, alpha_dice, cfg, epoch):
    model.train()
    running_loss = 0
    count = 0
    t0 = time.time()

    for bi, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x)["out"]
            ce = F.cross_entropy(out, y)
            dl = dice_loss(out, y)
            loss = ce + alpha_dice * dl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

        # LOGGING
        if bi % cfg.get("log_every_batches", 20) == 0 or bi == len(loader):
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / max(1, count)

            eta = (elapsed / bi) * (len(loader) - bi)
            safe_print(
                f"[Train E{epoch}] batch {bi}/{len(loader)} "
                f"loss={avg_loss:.4f} lr={lr:.2e} ETA={eta:.1f}s"
            )

        if cfg.get("max_batches_debug", 0) and bi >= cfg["max_batches_debug"]:
            break

    return running_loss / max(1, count)


# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, num_classes, cfg):
    model.eval()
    iou_sum = np.zeros(num_classes)
    iou_cnt = np.zeros(num_classes)

    total = 0
    t0 = time.time()

    for bi, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)["out"]      # B,C,H,W
        pred = out.argmax(1).cpu().numpy()
        tgt = y.cpu().numpy()

        for p, t in zip(pred, tgt):
            ious = iou_per_class_np(p, t, num_classes)
            for ci, v in enumerate(ious):
                if not np.isnan(v):
                    iou_sum[ci] += v
                    iou_cnt[ci] += 1

        total += pred.shape[0]

        if cfg.get("max_batches_debug", 0) and bi >= cfg["max_batches_debug"]:
            break

    per_class = []
    for c in range(num_classes):
        if iou_cnt[c] == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(iou_sum[c] / iou_cnt[c])

    mIoU = np.nanmean([v for v in per_class if not np.isnan(v)])
    safe_print(f"[Val] samples={total} time={time.time()-t0:.1f}s")

    return mIoU, per_class


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    safe_print(f"[INFO] Device: {device}")

    # Load stats
    mean = np.load(cfg["mean_npy"])
    std = np.load(cfg["std_npy"])
    train_list = cfg["train_list"]
    val_list   = cfg["val_list"]

    # Datasets
    train_ds = NpySegDataset(train_list, mean, std, augment=True,
                             debug_max_batches=cfg.get("max_batches_debug", 0))
    val_ds   = NpySegDataset(val_list, mean, std, augment=False,
                             debug_max_batches=cfg.get("max_batches_debug", 0))

    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                          num_workers=cfg.get("num_workers",4), pin_memory=True)

    val_dl   = DataLoader(val_ds, max(1,cfg["batch_size"]//2), shuffle=False,
                          num_workers=max(1,cfg.get("num_workers",2)), pin_memory=True)

    safe_print(f"[INFO] train samples: {len(train_ds)} batches: {len(train_dl)}")
    safe_print(f"[INFO] val   samples: {len(val_ds)} batches: {len(val_dl)}")

    # ------------------ Create model ------------------
    num_classes = cfg["num_classes"]
    try:
        from torchvision.models import ResNet50_Weights
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
            num_classes=num_classes
        )
        safe_print("[INFO] Loaded ResNet50 backbone (ImageNet).")
    except Exception as e:
        safe_print("[WARN] Could not load ResNet50 backbone weights:", e)
        model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
        safe_print("[INFO] Created DeepLabV3 with random init.")

    model = model.to(device)
    safe_print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------ OPTIMIZER ------------------
    opt_cfg = cfg.get("optimizer", {})
    try:
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-5))
    except:
        raise ValueError("Optimizer values must be numeric.")

    safe_print(f"[INFO] Optimizer AdamW lr={lr} weight_decay={wd}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    scheduler = None
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg.get("name") == "CosineAnnealingLR":
        T_max = int(sched_cfg.get("T_max", cfg.get("epochs", 30)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        safe_print(f"[INFO] Scheduler: CosineAnnealingLR (T_max={T_max})")

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ------------------ CHECKPOINTS ------------------
    ckpt_dir = Path(cfg.get("checkpoints_dir", "models"))
    ckpt_dir.mkdir(exist_ok=True)

    last_ckpt = ckpt_dir / cfg.get("checkpoint_last_name", "deeplabv3_pb_last.pth")
    best_ckpt = ckpt_dir / cfg.get("checkpoint_best_name", "deeplabv3_pb_best.pth")

    epoch_start = 0
    best_miou = -1

    # Resume if exists
    if last_ckpt.exists():
        safe_print(f"[RESUME] Loading checkpoint: {last_ckpt}")
        ck = torch.load(last_ckpt, map_location=device)
        try:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            epoch_start = ck.get("epoch", 0) + 1
            best_miou = ck.get("best_miou", -1)
            safe_print(f"[RESUME] Starting epoch={epoch_start}, best_mIoU={best_miou:.4f}")
        except Exception as e:
            safe_print("[WARN] Could not fully resume checkpoint:", e)

    # Save on Ctrl+C
    def on_signal(sig, frame):
        safe_print(f"\n[SIGNAL] Caught {sig}. Saving checkpoint...")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_start,
            "best_miou": best_miou
        }, last_ckpt)
        safe_print("[SIGNAL] Saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # ------------------ TRAIN LOOP ------------------
    epochs = cfg.get("epochs", 30)

    try:
        for epoch in range(epoch_start, epochs):
            safe_print("=" * 80)
            safe_print(f"[EPOCH {epoch+1}/{epochs}] starting")
            show_gpu_stats(device)

            train_loss = train_one_epoch(
                model, train_dl, optimizer, scaler, device,
                cfg.get("loss_alpha_dice", 0.5), cfg, epoch+1
            )

            safe_print(f"[EPOCH {epoch+1}] train_loss={train_loss:.4f}")

            # validation
            val_miou, per_class = validate(model, val_dl, device, num_classes, cfg)
            safe_print(f"[EPOCH {epoch+1}] val_mIoU={val_miou:.4f}")
            safe_print(" per-class:", ["{:.3f}".format(v) if not np.isnan(v) else "nan" for v in per_class])

            # update scheduler
            if scheduler:
                scheduler.step()

            # save last
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
            }, last_ckpt)

            # best
            if not math.isnan(val_miou) and val_miou > best_miou:
                best_miou = val_miou
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou
                }, best_ckpt)
                safe_print(f"[CHECKPOINT] New best saved (mIoU={best_miou:.4f})")

    except Exception as ex:
        safe_print("[ERROR] Exception during training:", ex)
        safe_print(traceback.format_exc())

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_start,
            "best_miou": best_miou
        }, last_ckpt)
        safe_print("[ERROR] Last checkpoint saved. Exiting.")
        raise

    safe_print("[TRAINING COMPLETE]")


if __name__ == "__main__":
    main()
