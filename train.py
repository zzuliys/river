import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2

from config import Config
from model import UNet
from dataset import get_dataloader

def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return ce_loss + d_loss

def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)

def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None, use_channels_last=False, non_blocking=False):
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        if use_channels_last:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
        else:
            images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.autocast(device_type='cuda', dtype=Config.AMP_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        iou = calculate_iou(pred, labels)
        total_iou += iou

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})

    return total_loss / len(dataloader), total_iou / len(dataloader)

def validate_epoch(model, dataloader, criterion, device, use_amp=False, use_channels_last=False, non_blocking=False):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            if use_channels_last:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
            else:
                images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=Config.AMP_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            iou = calculate_iou(pred, labels)
            total_iou += iou

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})

    return total_loss / len(dataloader), total_iou / len(dataloader)

def save_predictions(model, dataloader, device, save_dir, use_amp=False, use_channels_last=False, non_blocking=False):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:
                break
            if use_channels_last:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=non_blocking)
            else:
                images = images.to(device, non_blocking=non_blocking)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=Config.AMP_dtype):
                    outputs = model(images)
            else:
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            for j in range(images.size(0)):
                pred_mask = preds[j].cpu().numpy() * 255
                cv2.imwrite(os.path.join(save_dir, f'pred_{i}_{j}.png'), pred_mask)

def main():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    use_amp = Config.USE_AMP and device.type == 'cuda'
    use_channels_last = Config.USE_CHANNELS_LAST and device.type == 'cuda'
    non_blocking = Config.USE_NON_BLOCKING and device.type == 'cuda'
    if Config.CUDNN_BENCHMARK and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    if Config.USE_TF32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    print(f"Channels Last Memory Format: {'Enabled' if use_channels_last else 'Disabled'}")
    print(f"cudnn Benchmark: {'Enabled' if (Config.CUDNN_BENCHMARK and device.type == 'cuda') else 'Disabled'}")
    print(f"TF32: {'Enabled' if (Config.USE_TF32 and device.type == 'cuda') else 'Disabled'}")
    print(f"Non-Blocking Transfer: {'Enabled' if non_blocking else 'Disabled'}")
    print(f"Prefetch Factor: {Config.PREFETCH_FACTOR}")
    print(f"Persistent Workers: {'Enabled' if Config.PERSISTENT_WORKERS else 'Disabled'}")

    model = UNet(in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    use_compile = Config.USE_COMPILE and hasattr(torch, 'compile')
    if use_compile:
        model = torch.compile(model, mode="reduce-overhead")
    print(f"Model: U-Net with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"torch.compile: {'Enabled' if use_compile else 'Disabled'}")

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    train_loader = get_dataloader(
        Config.TRAIN_IMAGES,
        Config.TRAIN_LABELS,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
        Config.IMAGE_SIZE,
        shuffle=True,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    val_loader = get_dataloader(
        Config.VAL_IMAGES,
        Config.VAL_LABELS,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
        Config.IMAGE_SIZE,
        shuffle=False,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, fused=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_iou = 0
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")

        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler, use_channels_last, non_blocking)
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device, use_amp, use_channels_last, non_blocking)

        scheduler.step(val_iou)

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            print(f"Saved best model with IoU: {best_iou:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

    print("\nTraining completed!")
    print(f"Best IoU: {best_iou:.4f}")

    save_predictions(model, val_loader, device, Config.RESULT_DIR, use_amp, use_channels_last, non_blocking)
    print(f"Predictions saved to {Config.RESULT_DIR}")

if __name__ == "__main__":
    main()