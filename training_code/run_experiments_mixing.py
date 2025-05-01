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
import gc
from torchmetrics.functional import dice, average_precision  # Use torchmetrics
from torchmetrics.functional.segmentation import mean_iou 

# Set seeds for reproducibility
def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

seed_num = 42
seed_everything(seed_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if len(sys.argv) != 2:
    print("Usage: run_experiments_finetune.py <YAML_config_file>")
    sys.exit(1)

config_path = sys.argv[1]
config = load_config(config_path)
num_classes = config['num_classes']
target_res = config['target_res']
style = config['style']

######################################
#### Dataloading functions ###########
######################################
class ToBoolTensor(object):
    def __call__(self, tensor):
        return tensor.to(torch.bool)  # Convert to bool

class To32Tensor(object):
    def __call__(self, tensor):
        return tensor.to(torch.float32)

class To16Tensor(object):
    def __call__(self, tensor):
        return tensor.to(torch.float16)

class Rotate90():
    """Rotate the image by 90 degrees."""
    def __call__(self, x):
        return transforms.functional.rotate(x, angle=90, expand=True)

def convert_mask(mask_img, style, is_synthetic):
    mask_array = np.array(mask_img, dtype=np.int32)

    if style == 'White':
        # Binary segmentation combinding leaf and bark to a single class
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
        # 5 class segmentation where: 0: background/other/paper, 1: plant, 2: scalebar, 3: notes, 4: crc
        if is_synthetic:
            # New synthetic conversions for Milleville
            mask_array[mask_array == 5] = 0  # Paper to background/other
            mask_array[(mask_array == 150) | (mask_array == 250)] = 1  # Bark/leaf to plant
            mask_array[mask_array == 100] = 2  # Scale to scalebar
            mask_array[mask_array == 200] = 3  # Label to notes
            mask_array[mask_array == 50] = 4  # CRC to crc
        else:
            # Convert Milleville specific classes
            mask_array[(mask_array == 4) | (mask_array == 5) | (mask_array == 6)] = 3  # Barcode/stamp/attachment to notes
            mask_array[mask_array == 7] = 4  # color_card to crc
            mask_array[mask_array == 8] = 0  # Other to background/other
        mask_array[mask_array > 4] = 0 # anything not included in the first 5 classes is set to background/other

    mask = Image.fromarray(mask_array)
    return mask

# prep the data loader for the training data
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, src_transform=None, msk_transform=None,
                 msk_preprocess_func=None, is_synthetic=True, target_res=(640, 960), max_samples=0, augment=True, style='Hussein', num_classes=1, rotate=False):
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

        self.image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if '_src' in file])
        self.mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if '_msk' in file])
        
        if len(self.image_paths) > len(self.mask_paths):  # Address inconsistent naming
            self.mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if '_mask' in file])

        if max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.mask_paths = self.mask_paths[:max_samples]
            
        assert len(self.mask_paths) == len(self.image_paths), "Length of mask_paths does not match length of image_paths"

        self.total_samples = len(self.image_paths)
        self.height, self.width = target_res

        self.images_np, self.masks_np = self.load_images_and_masks()

    def load_images_and_masks(self):
        images_np = np.zeros((self.total_samples, 3, self.height, self.width), dtype=np.float32)
        masks_np = np.zeros((self.total_samples, self.height, self.width), dtype=np.int64)  # Use int64 for one-hot indices

        for idx, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
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
        image_np = self.images_np[idx].copy()
        mask_np = self.masks_np[idx].copy()

        image_tensor = torch.from_numpy(image_np)
        mask_tensor = torch.from_numpy(mask_np).long()  # Ensure labels are LongTensors

        if self.augment:
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [1])  # Flip width dimension
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, [1])
                mask_tensor = torch.flip(mask_tensor, [0])  # Flip height dimension

        return image_tensor, mask_tensor

    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

######################################
####### Training functions ###########
######################################
class DifferentiableDiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-6):
        super(DifferentiableDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        if self.num_classes == 2:
            # Binary segmentation
            inputs = torch.softmax(inputs, dim=1)[:, 1, ...]  # Probability of class 1
            inputs = inputs.reshape(-1)
            targets = targets.float().reshape(-1)  # Ensure targets are float
        else:
            # Multi-class segmentation
            inputs = torch.softmax(inputs, dim=1)  # Shape: [batch_size, num_classes, height, width]
            # One-hot encode targets
            targets_one_hot = nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            # Flatten inputs and targets
            inputs = inputs.view(inputs.size(0), self.num_classes, -1)
            targets = targets_one_hot.view(targets.size(0), self.num_classes, -1)

        # Compute Dice coefficient
        intersection = (inputs * targets).sum(dim=-1)
        union = inputs.sum(dim=-1) + targets.sum(dim=-1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()



def train_one_epoch(model, optimizer, data_loader, device, num_classes):
    model.train()
    scaler = GradScaler()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, disable=True)

    for step, (inputs, labels) in enumerate(progress_bar, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)['out']  # Shape: [batch_size, num_classes, height, width]

            # Calculate loss using the DifferentiableDiceLoss
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 10 == 0:
            progress_bar.set_description(f"Training step {step}, Loss: {loss.item():.4f}")

    return loss

def validate(model, data_loader, device, num_classes):
    model.eval()
    running_loss = 0.0

    dice_numerator = 0.0
    dice_denominator = 0.0
    miou_numerator = 0.0
    miou_denominator = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation", disable=True):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)['out']

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predictions
            if num_classes == 2:
                preds = torch.softmax(outputs, dim=1)[:, 1, ...]  # Probability of class 1
                preds = (preds > 0.5).long()
            else:
                preds = outputs.argmax(dim=1)

            # Compute metrics incrementally
            dice_score = dice(preds, labels, average="micro", num_classes=num_classes, ignore_index=0)
            miou_score = mean_iou(preds, labels, num_classes=num_classes, include_background=False)

            dice_numerator += dice_score * preds.size(0)
            dice_denominator += preds.size(0)

            miou_numerator += miou_score.mean() * preds.size(0)
            miou_denominator += preds.size(0)

            # Free up memory
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

    # Compute final metrics
    val_dice = dice_numerator / dice_denominator
    val_miou = miou_numerator / miou_denominator

    return running_loss / len(data_loader), val_dice.item(), val_miou.item()


def train_model(model, optimizer, train_loader, val_loader, scheduler, device, num_epochs=25, checkpoint_path='model_checkpoint.pt'):
    best_miou = 0.0

    for epoch in range(num_epochs):
        # Note train_loss is just the last batch's loss in this instance.
        train_loss = train_one_epoch(model, optimizer, train_loader, device, num_classes=config['num_classes'])
        val_loss, val_dice, val_miou = validate(model, val_loader, device, num_classes=config['num_classes'])
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val DICE: {val_dice:.4f}, Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_miou': val_miou,
            }, checkpoint_path)
            print(f"Model saved at epoch {epoch+1} with mIoU: {val_miou:.4f}")

        if scheduler:
            scheduler.step(val_loss)

    torch.cuda.empty_cache()

######################################
####### Inference functions ##########
######################################

def load_model_from_checkpoint(path, device='cuda'):
    config = load_config(config_path)
    num_classes = config['num_classes']
    weights = models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet101(weights=weights)
    num_input_features = model.classifier[4].in_channels
    new_classifier = nn.Conv2d(num_input_features, num_classes, kernel_size=(1, 1))
    model.classifier[4] = new_classifier
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(num_input_features, num_classes, kernel_size=(1, 1))
    model = model.to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def log_model_metrics(file_path, model_name, metrics):
    with open(file_path, 'a') as file:
        metrics_str = ", ".join([f"{key}: {value:.5f}" for key, value in metrics.items()])
        log_entry = f"Model: {model_name}, {metrics_str}\n"
        file.write(log_entry)

# Define model creation function
def create_model(num_classes):
    weights = models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet101(weights=weights)
    num_input_features = model.classifier[4].in_channels
    new_classifier = nn.Conv2d(num_input_features, num_classes, kernel_size=(1, 1))
    model.classifier[4] = new_classifier
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(num_input_features, num_classes, kernel_size=(1, 1))
    return model

def run_experiment(config,
                   gt_train_dset,
                   synth_dataset,
                   gt_val_dloader,
                   gt_test_dloader,
                   num_synth_samples,
                   num_real,
                   experiment_gt_name):

    target_res = config['target_res']
    b_size = config['batch_size']
    metric_log_file = config['metric_log_file']
    synth_train_src_dir = config['synth_train_src_dir']
    synth_train_mask_dir = config['synth_train_mask_dir']
    style = config['style']
    num_classes = config['num_classes']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']

    model = create_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    model.to(device)

    if gt_train_dset is not None:
        real_ds = gt_train_dset if num_real == len(gt_train_dset) else Subset(gt_train_dset, list(range(num_real)))
    else:
        real_ds = None

    if num_synth_samples > 0:  # when synthetic samples exist
        if num_synth_samples < len(synth_dataset):
            synth_ds = Subset(synth_dataset, list(range(num_synth_samples)))
        else:
            synth_ds = synth_dataset

        if real_ds:  # if there should be synthetic and real samples
            train_dataset = ConcatDataset([synth_ds, real_ds])
        else:
            train_dataset = synth_ds
    else:  # if no synthetic samples exist
        train_dataset = real_ds

    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
    checkpoint_path = f"{experiment_gt_name}.pt"
    train_model(model, optimizer,
                train_dataloader,
                gt_val_dloader,
                scheduler, device,
                num_epochs=num_epochs,
                checkpoint_path=checkpoint_path)

    # retrieve best model's metrics
    checkpoint = torch.load(checkpoint_path)
    metrics = {'epoc':checkpoint['epoch'],
               'val_loss':checkpoint['val_loss'],
               'val_miou':checkpoint['val_miou']}
    del checkpoint # purge it from ram
    
    # load best and log test metrics
    model, checkpoint = load_model_from_checkpoint(path=checkpoint_path)
    test_loss, test_dice, test_miou  = validate(model, gt_test_dloader, device, num_classes)
    print(f"{checkpoint_path} | Test Loss: {test_loss:.5f} | Test MIOU: {test_miou:.5f}")

    checkpoint.update({'test_dice': test_dice, 'test_loss': test_loss, 'test_miou': test_miou})
    # Save final model with test metrics
    torch.save(checkpoint, checkpoint_path)

    metrics['test_loss']= test_loss
    metrics['test_miou'] = test_miou
    log_model_metrics(metric_log_file, experiment_gt_name, metrics)

def has_completed_training(checkpoint_path):
    """Check if the model has completed training by verifying the presence of 'test_loss' in the checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'test_loss' in checkpoint:
            return True
    return False

# Define transforms
src_train_transform = transforms.Compose([
                                          transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
                                          transforms.ColorJitter(brightness=(0.9, 1.1),
                                                                 contrast=(0.9, 1.1),
                                                                 saturation=(0.9, 1.1),
                                                                 hue=(-0.01, 0.01)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                          To32Tensor()
                                          ])

msk_train_transform = transforms.Compose([
                                          transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
                                          ])

src_val_transform = transforms.Compose([transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        To32Tensor()
                                        ])

msk_val_transform = transforms.Compose([transforms.Resize(target_res, interpolation=transforms.InterpolationMode.NEAREST),
                                        ])

# Paths and dataset preparation
train_src_dir = config['train_src_dir']
train_mask_dir = config['train_mask_dir']
train_dataset = CustomDataset(image_dir=train_src_dir,
                                      mask_dir=train_mask_dir,
                                      src_transform=src_val_transform,
                                      msk_transform=msk_val_transform,
                                      msk_preprocess_func=convert_mask,
                                      is_synthetic=False,
                                      target_res=target_res,
                                      style=style,
                                      num_classes=num_classes)

print(f"{style} train dataset pre-loaded with {len(train_dataset)} samples")

validation_src_dir = config['validation_src_dir']
validation_mask_dir = config['validation_mask_dir']
validation_dataset = CustomDataset(image_dir=validation_src_dir,
                                   mask_dir=validation_mask_dir,
                                   src_transform=src_val_transform,
                                   msk_transform=msk_val_transform,
                                   is_synthetic=False,
                                   msk_preprocess_func=convert_mask,
                                   target_res=target_res,
                                   augment=False,
                                   style=style,
                                   num_classes=num_classes)

validation_dataloader = DataLoader(validation_dataset,
                                   batch_size=config['batch_size'],
                                   shuffle=False,
                                   pin_memory=True,
                                   drop_last=False,
                                   worker_init_fn=seed_worker)

print(f"{style} Validation data pre-loaded with {len(validation_dataset)} samples")

test_src_dir = config['test_src_dir']
test_mask_dir = config['test_mask_dir']
test_dataset = CustomDataset(image_dir=test_src_dir,
                             mask_dir=test_mask_dir,
                             src_transform=src_val_transform,
                             msk_transform=msk_val_transform,
                             msk_preprocess_func=convert_mask,
                             is_synthetic=False,
                             target_res=target_res,
                             augment=False,
                             style=style,
                             num_classes=num_classes)

test_dataloader = DataLoader(test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             worker_init_fn=seed_worker)
print(f"{style} Test data pre-loaded with {len(test_dataset)} samples")

synth_src_dir = config['synth_train_src_dir']
synth_mask_dir = config['synth_train_mask_dir']
synth_dataset = CustomDataset(image_dir=synth_src_dir,
                              mask_dir=synth_mask_dir,
                              src_transform=src_train_transform,
                              msk_transform=msk_train_transform,
                              msk_preprocess_func=convert_mask,
                              is_synthetic=True,
                              target_res=target_res,
                              style=style,
                              num_classes=num_classes,
                              max_samples=max(config['num_synth_samples']))

print(f"Synthetic dataset pre-loaded with {len(synth_dataset)} samples")

# Configurations and run experiments
metric_log_file = config['metric_log_file']
real_proportions = config['real_proportions']
max_real_samples = config['max_real_samples']
run_name = config['run_name']
num_synth_samples = config['num_synth_samples'] # refresh list of synthetic samples to test    

criterion = DifferentiableDiceLoss(num_classes=config['num_classes'])

for real_proportion in real_proportions:
    
    num_real = int(round(real_proportion * max_real_samples))
    real_ds = None if num_real == 0 else Subset(train_dataset, list(range(num_real)))

    # Prep the synthetic test dataset sizes to include 1:1 experiments
    #num_synth_samples.append(max_real_samples) # Include synthetic samples equal to the original real size
    #list(set(num_synth_samples)).sort()
    for num_synth_sample in num_synth_samples:
        if num_real == 0 and num_synth_sample == 0:  # skip the 0:0 composition
            continue
        
        experiment_gt_name = f"{run_name}_{num_real}_real_{num_synth_sample}_synth"
        if has_completed_training(f"{experiment_gt_name}.pt"):
            print(f"checkpoint {experiment_gt_name}.pt completed, skipping")
            continue

        print(f"Testing on {num_real} real samples, {num_synth_sample} synthetic samples")
        seed_everything(seed_num)  # Ensure each network gets the same beginning. 
        run_experiment(config=config,
                       gt_train_dset=real_ds,
                       synth_dataset = synth_dataset,
                       gt_val_dloader=validation_dataloader,
                       gt_test_dloader=test_dataloader,
                       num_synth_samples=num_synth_sample,
                       num_real = num_real,
                       experiment_gt_name=experiment_gt_name)
