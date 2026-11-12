        # 5 class segmentation where:
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from tqdm import tqdm
import random
import os
from PIL import Image
import numpy as np
import yaml
import sys

from torchmetrics.functional import dice
from torchmetrics.functional.segmentation import mean_iou


# ─── Reproducibility ───────────────────────────────────────────────────────────────

def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


seed_num = 42
seed_everything(seed_num)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Config ────────────────────────────────────────────────────────────────────────

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if len(sys.argv) != 2:
    print("Usage: run_experiments_mixing_v3.py <YAML_config_file>")
    sys.exit(1)

config_path = sys.argv[1]
config = load_config(config_path)
num_classes = config['num_classes']
target_res = config['target_res']
style = config['style']


# ─── Class names (for readable log output) ────────────────────────────────────────

CLASS_NAMES = {
    'White':      ['background', 'leaf_stem'],
    'Hussein':    ['background', 'leaf'],
    'Milleville': ['background', 'plant', 'scalebar', 'notes', 'crc'],
}


# ─── Data loading utilities ────────────────────────────────────────────────────────

class ToBoolTensor:
    def __call__(self, tensor):
        return tensor.to(torch.bool)


class To32Tensor:
    def __call__(self, tensor):
        return tensor.to(torch.float32)


class Rotate90:
    """Rotate image/mask by 90 degrees with expansion."""
    def __call__(self, x):
        return transforms.functional.rotate(x, angle=90, expand=True)


def convert_mask(mask_img, style, is_synthetic):
    mask_array = np.array(mask_img, dtype=np.int32)

    if style == 'White':
        # Binary segmentation combining leaf and stem to a single class
        if is_synthetic:
            mask_array[(mask_array == 150) | (mask_array == 250)] = 1
        else:
            mask_array[mask_array > 0] = 1  # anything not 0 is 255
        mask_array[mask_array != 1] = 0

    elif style == 'Hussein':
        # Binary segmentation only selecting for the leaves
        if is_synthetic:
            mask_array[mask_array == 250] = 1
        else:
            mask_array[mask_array > 0] = 1  # anything not 0 is 255
        mask_array[mask_array != 1] = 0

    elif style == 'Milleville':
        # 5 class segmentation where:
        #   0: background/other/paper, 1: plant, 2: scalebar, 3: notes, 4: crc
        if is_synthetic:
            # New synthetic conversions for Milleville
            mask_array[mask_array == 5] = 0                          # Paper → background/other
            mask_array[(mask_array == 150) | (mask_array == 250)] = 1  # stem/leaf → plant
            mask_array[mask_array == 100] = 2                        # Scale → scalebar
            mask_array[mask_array == 200] = 3                        # Label → notes
            mask_array[mask_array == 50] = 4                         # CRC → crc
        else:
            # Convert Milleville specific classes
            mask_array[(mask_array == 4) | (mask_array == 5) | (mask_array == 6)] = 3  # Barcode/stamp/attachment → notes
            mask_array[mask_array == 7] = 4  # color_card → crc
            mask_array[mask_array == 8] = 0  # Other → background/other
        mask_array[mask_array > 4] = 0  # anything not in the first 5 classes → background/other

    return Image.fromarray(mask_array)


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, src_transform=None, msk_transform=None,
                 msk_preprocess_func=None, is_synthetic=True, target_res=(640, 960),
                 max_samples=0, augment=True, style='Hussein', num_classes=1, rotate=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.src_transform = src_transform
        self.msk_transform = msk_transform
        self.msk_preprocess_func = msk_preprocess_func
        self.is_synthetic = is_synthetic
        self.augment = augment
        self.style = style
        self.num_classes = num_classes
        self.rotate = rotate

        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if '_src' in f]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if '_msk' in f]
        )

        # Fallback naming convention (some datasets use _mask instead of _msk)
        if len(self.image_paths) > len(self.mask_paths):
            self.mask_paths = sorted(
                [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if '_mask' in f]
            )

        if max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]

        assert len(self.mask_paths) == len(self.image_paths), \
            "Mask/image path count mismatch"

        self.total_samples = len(self.image_paths)
        self.height, self.width = target_res
        self.images_np, self.masks_np = self._load_all()

    def _load_all(self):
        images_np = np.zeros((self.total_samples, 3, self.height, self.width), dtype=np.float32)
        masks_np = np.zeros((self.total_samples, self.height, self.width), dtype=np.int64)

        for idx, (img_path, msk_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(msk_path).convert("L")

            if self.rotate:
                image = Rotate90()(image)
                mask = Rotate90()(mask)
            if self.src_transform:
                image = self.src_transform(image)
            if self.msk_transform:
                mask = self.msk_preprocess_func(mask, self.style, is_synthetic=self.is_synthetic)
                mask = self.msk_transform(mask)

            images_np[idx] = np.array(image)
            masks_np[idx] = np.array(mask)

        return images_np, masks_np

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        image_tensor = torch.from_numpy(self.images_np[idx].copy())
        mask_tensor = torch.from_numpy(self.masks_np[idx].copy()).long()

        if self.augment:
            # Random horizontal flip (dim 2 of CHW tensor)
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [1])
            # Random vertical flip (dim 1 of CHW tensor)
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, [1])
                mask_tensor = torch.flip(mask_tensor, [0])

        return image_tensor, mask_tensor


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ─── Class weight computation (fixed across all ablation arms) ─────────────────────

def compute_class_weights(masks, num_classes):
    """Compute inverse-frequency class weights from the full real training set.

    Normalized to mean 1 so they rescale the loss without changing relative
    weighting. Class 0 receives a weight but is excluded via ignore_index=0.

    Args:
        masks: np.ndarray [N, H, W] of int labels (full real training set).
        num_classes: total number of classes including background.

    Returns:
        torch.FloatTensor [num_classes] on device.
    """
    flat = masks.flatten()
    counts = np.bincount(flat.astype(np.int64), minlength=num_classes).astype(np.float64)
    # Floor at 1 to avoid division by zero for absent classes
    counts = np.maximum(counts, 1.0)
    weights = len(flat) / (num_classes * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ─── Training / validation ─────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, disable=False)
    running_loss = 0.0

    for step, (inputs, labels) in enumerate(progress_bar, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            running_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            progress_bar.set_description(f"Train step {step}, Loss: {loss.item():.4f}")

    return running_loss / len(data_loader)


def validate(model, data_loader, device, num_classes):
    """Evaluate model and accumulate per-class segmentation metrics.

    Returns dict with:
        loss          – mean CE over the dataset (accounting for partial last batch)
        dice_micro    – pooled Dice across all foreground classes
        dice_macro    – mean of per-class Dice over foreground classes
        miou          – mean Intersection-over-Union over foreground classes
        iou_per_class – FloatTensor [num_classes], IoU for EVERY class including bg
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    # Accumulate TP/FP/FN for ALL classes (including background) so we can
    # report per-class IoU for every class. Summary metrics use fg only.
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, labels in data_loader:
            batch_size = inputs.size(0)
            n_samples += batch_size

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)['out']
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size  # accumulate un-averaged sum

            # Hard predictions matching inference decision rule
            if num_classes == 2:
                preds = (torch.softmax(outputs, dim=1)[:, 1] > 0.5).long()
            else:
                preds = outputs.argmax(dim=1)

            # Per-class TP/FP/FN accumulation (all classes)
            for c in range(num_classes):
                pred_c = (preds == c)
                target_c = (labels == c)
                tp[c] += (pred_c & target_c).sum()
                fp[c] += (pred_c & ~target_c).sum()
                fn[c] += (~pred_c & target_c).sum()

    # Normalize loss to per-sample mean
    mean_loss = running_loss / max(n_samples, 1)

    # Per-class IoU for ALL classes (includes background at index 0)
    iou_per_class = tp / (tp + fp + fn + 1e-8)

    # Summary metrics computed on foreground only (classes 1..C-1)
    fg_tp, fg_fp, fg_fn = tp[1:], fp[1:], fn[1:]
    dice_micro = (2 * fg_tp.sum()) / (2 * fg_tp.sum() + fg_fp.sum() + fg_fn.sum() + 1e-8)
    iou_fg = fg_tp / (fg_tp + fg_fp + fg_fn + 1e-8)
    miou = iou_fg.mean()
    dice_macro = ((2 * fg_tp) / (2 * fg_tp + fg_fp + fg_fn + 1e-8)).mean()

    return {
        'loss': mean_loss,
        'dice_micro': dice_micro.item(),
        'dice_macro': dice_macro.item(),
        'miou': miou.item(),
        'iou_per_class': iou_per_class.cpu(),  # shape [num_classes], index 0 = bg
    }


def train_model(model, optimizer, scheduler, train_loader, val_loader, device,
                num_epochs=50, checkpoint_path='model_checkpoint.pt'):
    """Train for a fixed number of epochs. Best model selected on val mIoU."""
    best_miou = 0.0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_metrics = validate(model, val_loader, device, num_classes)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val mIoU: {val_metrics['miou']:.4f} | "
            f"Val DICE(micro): {val_metrics['dice_micro']:.4f}"
        )

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': {k: v for k, v in val_metrics.items()},
            }, checkpoint_path)
            print(f"  \u2192 Best saved (mIoU={best_miou:.4f})")

        if scheduler is not None:
            scheduler.step(val_metrics['loss'])

    # Final-epoch metrics for convergence reporting
    last_val = validate(model, val_loader, device, num_classes)
    return best_miou, last_val


# ─── Model creation / loading ──────────────────────────────────────────────────────

def create_model(num_classes):
    """DeepLabV3 + ResNet101 with COCO-pretrained backbone, fresh segmentation head."""
    weights = models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet101(weights=weights)
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    return model


def load_model_from_checkpoint(path, device='cuda'):
    """Load trained checkpoint for test evaluation.

    Builds with COCO weights then overwrites via state_dict (same as v2).
    Kept for parity to avoid introducing a new failure mode.
    """
    cfg = load_config(config_path)
    nc = cfg['num_classes']
    model = create_model(nc)
    model.to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ─── Logging ───────────────────────────────────────────────────────────────────────

def log_model_metrics(file_path, model_name, metrics):
    """Append one row to the results file.

    `metrics` is an ordered dict of {key: float | str}.
    Floats → 5 decimal places. Strings written verbatim.
    """
    parts = []
    for key, value in metrics.items():
        if isinstance(value, str):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: {value:.5f}")
    line = f"Model: {model_name}, " + ", ".join(parts)
    with open(file_path, 'a') as f:
        f.write(line + "\n")


# ─── Resume check ──────────────────────────────────────────────────────────────────

def has_completed_training(checkpoint_path):
    """Return True if the checkpoint file exists and contains test metrics."""
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        if 'test_metrics' in ckpt:
            return True
    return False


# ─── Experiment runner ─────────────────────────────────────────────────────────────

def run_experiment(config, real_indices, synth_dataset, gt_val_loader, gt_test_loader,
                   num_synth_samples, class_weights_tensor, experiment_name):
    """Train one model for a given (real_count, synth_count) combination."""
    b_size = config['batch_size']
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    class_names = CLASS_NAMES[style]

    # Build training dataset from selected real indices + synthetic subset
    train_ds_real = Subset(train_dataset, real_indices) if len(real_indices) > 0 else None

    if num_synth_samples > 0:
        synth_subset = (
            Subset(synth_dataset, list(range(num_synth_samples)))
            if num_synth_samples < len(synth_dataset)
            else synth_dataset
        )
        if train_ds_real is not None:
            full_train = ConcatDataset([synth_subset, train_ds_real])
        else:
            full_train = synth_subset
    else:
        full_train = train_ds_real

    train_loader = DataLoader(
        full_train, batch_size=b_size, shuffle=True,
        pin_memory=True, drop_last=True, worker_init_fn=seed_worker
    )

    # Model + optimizer + scheduler
    model = create_model(num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )

    # Train (fixed epochs for cross-arm comparability)
    ckpt_path = f"{experiment_name}.pt"
    best_miou, last_val = train_model(
        model, optimizer, scheduler, train_loader, gt_val_loader, device,
        num_epochs=num_epochs, checkpoint_path=ckpt_path
    )

    # Free training resources before reloading for test eval
    del model, optimizer, scheduler, train_loader, full_train
    torch.cuda.empty_cache()

    # Load best model and evaluate on held-out test set
    model, ckpt = load_model_from_checkpoint(ckpt_path)
    test_metrics = validate(model, gt_test_loader, device, num_classes)

    print(
        f"\n{experiment_name} | "
        f"Test Loss: {test_metrics['loss']:.5f} | "
        f"Test mIoU: {test_metrics['miou']:.5f} | "
        f"Test DICE(micro): {test_metrics['dice_micro']:.5f}"
    )

    # ── Build log entry with named per-class metrics ──
    cw_str = "[" + ", ".join(f"{w:.3f}" for w in class_weights_tensor.cpu().tolist()) + "]"
    val_m = ckpt['val_metrics']

    metrics_ordered = {}

    # Summary
    metrics_ordered['best_epoch'] = float(ckpt['epoch'])
    metrics_ordered['train_loss_best'] = ckpt['train_loss']

    # Validation at best epoch
    metrics_ordered['val_loss'] = val_m['loss']
    metrics_ordered['val_dice_micro'] = val_m['dice_micro']
    metrics_ordered['val_dice_macro'] = val_m['dice_macro']
    metrics_ordered['val_miou'] = val_m['miou']

    # Per-class IoU at validation (all classes including background)
    for i in range(num_classes):
        name = class_names[i]
        metrics_ordered[f'val_iou_{name}'] = val_m['iou_per_class'][i].item()

    # Last-epoch validation (convergence diagnostic)
    metrics_ordered['last_val_loss'] = last_val['loss']
    metrics_ordered['last_val_miou'] = last_val['miou']

    # Test set summary
    metrics_ordered['test_loss'] = test_metrics['loss']
    metrics_ordered['test_dice_micro'] = test_metrics['dice_micro']
    metrics_ordered['test_dice_macro'] = test_metrics['dice_macro']
    metrics_ordered['test_miou'] = test_metrics['miou']

    # Per-class IoU at test (all classes including background)
    for i in range(num_classes):
        name = class_names[i]
        metrics_ordered[f'test_iou_{name}'] = test_metrics['iou_per_class'][i].item()

    # Class weights used (identical across all arms; logged for self-contained table)
    metrics_ordered['class_weights'] = cw_str

    log_model_metrics(config['metric_log_file'], experiment_name, metrics_ordered)

    # Save test metrics into checkpoint for resume safety
    ckpt['test_metrics'] = {
        k: v for k, v in test_metrics.items() if not isinstance(v, torch.Tensor)
    }
    ckpt['test_iou_per_class'] = test_metrics['iou_per_class'].tolist()
    torch.save(ckpt, ckpt_path)


# ─── Transforms ────────────────────────────────────────────────────────────────────

src_train_transform = transforms.Compose([
    transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ColorJitter(
        brightness=(0.9, 1.1), contrast=(0.9, 1.1),
        saturation=(0.9, 1.1), hue=(-0.01, 0.01)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    To32Tensor()
])

src_val_transform = transforms.Compose([
    transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    To32Tensor()
])

msk_transform = transforms.Compose([
    transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
])


# ─── Dataset instantiation ─────────────────────────────────────────────────────────

print(f"Loading {style} training set...")
train_dataset = CustomDataset(
    image_dir=config['train_src_dir'],
    mask_dir=config['train_mask_dir'],
    src_transform=src_val_transform,  # aug applied in __getitem__, not at load time
    msk_transform=msk_transform,
    msk_preprocess_func=convert_mask,
    is_synthetic=False,
    target_res=target_res,
    augment=True,
    style=style,
    num_classes=num_classes,
)
print(f"  {len(train_dataset)} real training samples")

# Fixed class weights from full real set (held constant across all ablation arms)
class_weights_tensor = compute_class_weights(train_dataset.masks_np, num_classes)
print(f"  Class weights: {class_weights_tensor.cpu().tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"Loading {style} validation set...")
val_dataset = CustomDataset(
    image_dir=config['validation_src_dir'],
    mask_dir=config['validation_mask_dir'],
    src_transform=src_val_transform,
    msk_transform=msk_transform,
    msk_preprocess_func=convert_mask,
    is_synthetic=False,
    target_res=target_res,
    augment=False,
    style=style,
    num_classes=num_classes,
)
val_loader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False,
    pin_memory=True, drop_last=False, worker_init_fn=seed_worker
)
print(f"  {len(val_dataset)} validation samples")

print(f"Loading {style} test set...")
test_dataset = CustomDataset(
    image_dir=config['test_src_dir'],
    mask_dir=config['test_mask_dir'],
    src_transform=src_val_transform,
    msk_transform=msk_transform,
    msk_preprocess_func=convert_mask,
    is_synthetic=False,
    target_res=target_res,
    augment=False,
    style=style,
    num_classes=num_classes,
)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False,
    pin_memory=True, drop_last=False, worker_init_fn=seed_worker
)
print(f"  {len(test_dataset)} test samples")

synth_src_dir = config['synth_train_src_dir']
synth_mask_dir = config['synth_train_mask_dir']
synth_dataset = CustomDataset(
    image_dir=synth_src_dir,
    mask_dir=synth_mask_dir,
    src_transform=src_train_transform,
    msk_transform=msk_transform,
    msk_preprocess_func=convert_mask,
    is_synthetic=True,
    target_res=target_res,
    augment=True,
    style=style,
    num_classes=num_classes,
    max_samples=max(config['num_synth_samples']),
)
print(f"  {len(synth_dataset)} synthetic samples loaded")


# ─── Experiment loop ───────────────────────────────────────────────────────────────

real_proportions = config['real_proportions']
max_real_samples = config['max_real_samples']
run_name = config['run_name']
num_synth_list = list(config['num_synth_samples'])

# Fixed random permutation for reproducible subsampling.
# Smaller arms are subsets of larger ones, but selection is independent of
# filename ordering (avoids bias if filenames encode species/collector/etc.).
full_real_indices = torch.randperm(len(train_dataset)).tolist()

for real_proportion in real_proportions:
    num_real = int(round(real_proportion * max_real_samples))
    real_indices = full_real_indices[:num_real] if num_real > 0 else []

    for num_synth in num_synth_list:
        if num_real == 0 and num_synth == 0:
            continue

        experiment_name = f"{run_name}_{num_real}_real_{num_synth}_synth"

        # Resume safety: skip if test metrics already saved to checkpoint
        ckpt_file = f"{experiment_name}.pt"
        if has_completed_training(ckpt_file):
            print(f"\n[SKIP] {experiment_name} \u2013 completed, moving on.")
            continue

        print(f"\n{'='*60}")
        print(f"  {num_real} real + {num_synth} synthetic")
        print(f"{'='*60}")

        seed_everything(seed_num)  # identical weight init across all arms

        run_experiment(
            config=config,
            real_indices=real_indices,
            synth_dataset=synth_dataset,
            gt_val_loader=val_loader,
            gt_test_loader=test_loader,
            num_synth_samples=num_synth,
            class_weights_tensor=class_weights_tensor,
            experiment_name=experiment_name,
        )

print("\nAll experiments complete.")
