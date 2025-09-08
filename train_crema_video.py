import argparse
import datetime
import logging
import os
import random
import signal
import sys
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2
from spikingjelly.activation_based import functional
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import (AverageMeter, accuracy, setup_default_logging, NativeScaler, dispatch_clip_grad, get_outdir, )
from torch.utils.data import DataLoader

from datasets.crema.crema_video import CremaVideo
from models.video_model import HierarchicalSpikingTransformer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def create_transforms(args, is_training=True):
    """
    Create transforms using torchvision.transforms.v2.
    """
    # Grayscale normalization parameters (single channel)
    mean = [0.485]  # Single channel mean
    std = [0.229]  # Single channel std

    if is_training:
        # --- MODIFIED AUGMENTATION PIPELINE ---
        # Added RandomErasing instead of cutmix/mixup for regularization.
        transforms_list = [transforms_v2.ToDtype(torch.float32, scale=True),
                           transforms_v2.ColorJitter(brightness=0.2, contrast=0.2),
                           transforms_v2.RandomRotation(degrees=5),
                           transforms_v2.RandomHorizontalFlip(p=0.5),
                           transforms_v2.Normalize(mean=mean, std=std)]

        # Add RandomErasing if enabled
        if args.reprob > 0:
            transforms_list.append(
                transforms_v2.RandomErasing(p=args.reprob, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0,
                                            inplace=False))

        return transforms_v2.Compose(transforms_list)
    else:
        # Standard validation transforms
        return transforms_v2.Compose(
            [transforms_v2.ToDtype(torch.float32, scale=True), transforms_v2.Normalize(mean=mean, std=std), ])


def create_dataloaders(args, _logger):
    """Create train, validation, and test dataloaders using CREMA."""
    test_transform = create_transforms(args, is_training=False)
    val_transform = create_transforms(args, is_training=False)

    # Create test dataset
    test_dataset = CremaVideo(data_path=args.data_path, num_frames=args.time_step, frame_size=args.img_size,
                              split="test", transform=test_transform, )

    # Create validation dataset
    val_dataset = CremaVideo(data_path=args.data_path, num_frames=args.time_step, frame_size=args.img_size, split="val",
                             transform=val_transform, )

    _logger.info("Using standard single augmentation strategy for training.")
    train_transform = create_transforms(args, is_training=True)
    train_dataset = CremaVideo(data_path=args.data_path, num_frames=args.time_step, frame_size=args.img_size,
                               split="train", transform=train_transform, )

    _logger.info(
        f"Using CREMA dataset. Found {len(train_dataset)} training samples, {len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, )
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, )
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, )
    return train_loader, val_loader, test_loader


def create_mixup_fn(args, _logger):
    """Create MixUp and CutMix transforms using v2 API based on timm args."""
    mixup_active = args.mixup > 0 or args.cutmix > 0
    if not mixup_active:
        return None

    mixup_transforms = []
    if args.mixup > 0:
        mixup_transforms.append(transforms_v2.MixUp(alpha=args.mixup, num_classes=args.num_classes))
    if args.cutmix > 0:
        mixup_transforms.append(transforms_v2.CutMix(alpha=args.cutmix, num_classes=args.num_classes))

    if len(mixup_transforms) == 1:
        _logger.info(
            f"Using {mixup_transforms[0].__class__.__name__} with alpha={getattr(mixup_transforms[0], 'alpha', 'N/A')}")
        return mixup_transforms[0]
    elif len(mixup_transforms) == 2:
        probs = [1.0 - args.mixup_switch_prob, args.mixup_switch_prob]
        _logger.info(f"Using RandomChoice between MixUp (p={probs[0]}) and CutMix (p={probs[1]})")
        return transforms_v2.RandomChoice(mixup_transforms, p=probs)
    else:
        return None


def train_one_epoch(epoch, model, loader, optimizer, criterion_soft, criterion_hard, args, lr_scheduler=None,
                    loss_scaler=None, mixup_fn=None, ):
    model.train()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    epoch_start_time = time.time()
    end = time.time()
    num_updates = epoch * len(loader)

    use_mixup = mixup_fn is not None and epoch < args.mixup_off_epoch

    def _apply_mixup_safe(images, labels):
        if images.ndim == 5:
            B, T, C, H, W = images.shape
            images_merged = images.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
            mixed_images, mixed_labels = mixup_fn(images_merged, labels)
            mixed_images = mixed_images.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
            return mixed_images, mixed_labels
        else:
            return mixup_fn(images, labels)

    for batch_idx, (images, labels) in enumerate(loader):
        data_time_m.update(time.time() - end)
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        if use_mixup:
            images, labels = _apply_mixup_safe(images, labels)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            output = model(images)
            loss = criterion_soft(output, labels) if use_mixup else criterion_hard(output, labels)

        acc_labels = labels.max(dim=1)[1] if labels.ndim == 2 else labels
        acc1, _ = accuracy(output, acc_labels, topk=(1, 5))
        losses_m.update(loss.item(), images.size(0))
        top1_m.update(acc1.item(), images.size(0))

        optimizer.zero_grad()
        if loss_scaler:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
        else:
            loss.backward()
            if args.clip_grad:
                dispatch_clip_grad(model.parameters(), value=args.clip_grad, mode="norm")
            optimizer.step()

        functional.reset_net(model)
        torch.cuda.synchronize()

        num_updates += 1
        end = time.time()

        if lr_scheduler:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

    return OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("duration", time.time() - epoch_start_time),
                        ("data_time", data_time_m.avg), ])


@torch.no_grad()
def evaluate(model, loader, loss_fn, args):
    model.eval()
    losses_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    batch_time_m = AverageMeter()

    epoch_start_time = time.time()
    last_idx = len(loader) - 1
    end = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        functional.reset_net(model)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        losses_m.update(loss.item(), images.size(0))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

    epoch_duration = time.time() - epoch_start_time
    return OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg), ("duration", epoch_duration), ])


@torch.no_grad()
def test_final_model(model, test_loader, loss_fn, args, _logger):
    """Evaluate the final model on the test set."""
    _logger.info("Starting final test evaluation...")
    model.eval()
    losses_m, top1_m, top5_m = AverageMeter(), AverageMeter(), AverageMeter()
    batch_time_m = AverageMeter()

    test_start_time = time.time()
    last_idx = len(test_loader) - 1
    end = time.time()

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        functional.reset_net(model)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        losses_m.update(loss.item(), images.size(0))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0 or batch_idx == last_idx:
            _logger.info(f"Test batch [{batch_idx}/{last_idx}] - Acc@1: {acc1.item():.2f}% Acc@5: {acc5.item():.2f}%")

    test_duration = time.time() - test_start_time
    _logger.info(
        f"Final Test Results - Loss: {losses_m.avg:.4f} | Acc@1: {top1_m.avg:.2f}% | Acc@5: {top5_m.avg:.2f}% | Duration: {test_duration:.2f}s")

    return OrderedDict(
        [("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg), ("duration", test_duration), ])


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch SNN Training with Imagenette")

    # --- Paths and Basic Config ---
    parser.add_argument("--data-path", type=str,
                        default="/user/mohamed.saleh01/u18697/projects/datasets/crema/Crema_frams_80_10_10",
                        help="Path to dataset root folder (should contain train, val, and test subdirectories)", )
    parser.add_argument("--output-dir", type=str, default="./out/crema/video",
                        help="Path to save checkpoints and logs", )
    parser.add_argument("--model", default="qk_video", type=str, help="Name of model to train")
    parser.add_argument("--resume", default="", type=str, help="Path to latest checkpoint")
    parser.add_argument("--epochs", type=int, default=180, help="Number of base epochs to train")
    parser.add_argument("--start-epoch", type=int, default=0, help="Manual epoch number")
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--val-batch-size", type=int, default=32, help="Input batch size for validation")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--amp", action="store_true", default=True, help="Use PyTorch Native AMP")

    # --- Early Stopping ---
    parser.add_argument("--patience-epochs", type=int, default=25,
                        help="Patient epochs for early stopping. Set to 0 to disable.", )

    # --- Plot Update Interval ---
    parser.add_argument("--plot-update-interval", type=int, default=30,
                        help="Update plots every N epochs during training", )

    # --- Model Architecture ---
    parser.add_argument("--num-classes", type=int, default=6, help="Number of label classes (for CREMA-D)", )
    parser.add_argument("--img-size", type=int, default=128, help="Input image size")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[2, 4, 8], help="Attention head number", )
    parser.add_argument("--embed-dims", type=int, default=512, help="Embedd8ing dimension")
    parser.add_argument("--mlp-ratios", type=int, default=4, help="MLP hidden dimension ratio")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8], help="Number of transformer blocks", )
    parser.add_argument("--time-step", "-T", type=int, default=8, help="Simulation time steps for SNN")

    # --- Optimizer ---
    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--clip-grad", type=float, default=1, help="Clip gradient norm")
    parser.add_argument("--opt-eps", default=None, type=float, help="Optimizer Epsilon")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", help="Optimizer Betas")

    # --- LR Scheduler ---
    parser.add_argument("--sched", default="cosine", type=str, help="LR scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Epochs to warmup LR")
    parser.add_argument("--warmup-lr", type=float, default=1e-6, help="Warmup learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for LR")
    parser.add_argument("--cooldown-epochs", type=int, default=10, help="Cooldown epochs")

    # --- Data Augmentation & Regularization (MODIFIED FOR FACIAL RECOGNITION) ---
    parser.add_argument("--mean", type=float, nargs="+", default=[0.485, 0.456, 0.406],
                        help="Dataset mean (ImageNet default)", )
    parser.add_argument("--std", type=float, nargs="+", default=[0.229, 0.224, 0.225],
                        help="Dataset std (ImageNet default)", )
    parser.add_argument("--crop-pct", type=float, default=1.0, help="Input image center crop percent")
    parser.add_argument("--scale", type=float, nargs="+", default=[0.9, 1.0],
                        help="Random resize scale (more conservative)", )
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.9, 1.1], help="Random resize aspect ratio", )
    parser.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip probability")
    parser.add_argument("--vflip", type=float, default=0.0, help="Vertical flip probability")
    parser.add_argument("--color-jitter", type=float, default=0.2, help="Color jitter factor (reduced)")
    parser.add_argument("--aa", type=str, default="rand-m5-mstd0.5-inc1",
                        help="AutoAugment policy (reduced magnitude)", )
    parser.add_argument("--train-interpolation", type=str, default="bicubic", help="Training interpolation", )
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--reprob", type=float, default=0.9, help="Random erase probability")
    parser.add_argument("--remode", type=str, default="pixel", help="Random erase mode")
    parser.add_argument("--recount", type=int, default=1, help="Random erase count")
    parser.add_argument("--resplit", action="store_true", help="Random erase per-sample")

    # --- Mixup (MODIFIED) ---
    # Increased alpha values for stronger regularization
    parser.add_argument("--mixup", type=float, default=0.3, help="Mixup alpha (increased)")
    parser.add_argument("--cutmix", type=float, default=0.0, help="Cutmix alpha (increased)")
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5, help="Probability of switching to cutmix", )
    parser.add_argument("--mixup-off-epoch", default=1000, type=int, help="Turn off mixup after this epoch", )

    # --- SNN Neuron Config ---
    parser.add_argument("--neuron-type", type=str, default="PLIF", help="Type of neuron to use (e.g., LIF)", )
    parser.add_argument("--tau", type=float, default=2.0, help="Membrane time constant")
    parser.add_argument("--v_threshold", type=float, default=1.0, help="Voltage threshold")
    parser.add_argument("--surrogate-function", type=str, default="sigmoid", help="Type of surrogate function", )

    return parser.parse_args()


def main():
    args = get_args()

    # Setup logging
    output_dir = get_outdir(args.output_dir,
                            str(args.neuron_type) + "_" + str(args.surrogate_function) + "_" + str(
                                args.time_step) + "_" + f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}", )
    log_file_path = os.path.join(output_dir, time.strftime("%Y_%m_%d_%H_%M_%S.log"))
    setup_default_logging(log_path=log_file_path)
    _logger = logging.getLogger(__name__)

    # Global variables for signal handler
    global train_accuracies, val_accuracies, train_losses, val_losses, global_output_dir, global_logger
    train_accuracies, val_accuracies, train_losses, val_losses = [], [], [], []
    global_output_dir = output_dir
    global_logger = _logger

    def signal_handler(signum, frame):
        """Handle script interruption by saving current plots."""
        _logger.info(f"\nReceived signal {signum}. Saving plots and exiting gracefully...")
        if train_accuracies or val_accuracies:
            plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, output_dir, _logger)
        sys.exit(0)

    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # Log all arguments for reproducibility
    _logger.info("-----------------")
    _logger.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        _logger.info(f"- {arg}: {value}")
    _logger.info("-----------------")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(f"Training on device: {device}")

    model = HierarchicalSpikingTransformer(embed_dims=args.embed_dims, num_heads=args.num_heads,
                                           mlp_ratios=args.mlp_ratios, in_channels=1, num_classes=args.num_classes,
                                           depths=args.depths, T=args.time_step, neuron_type=args.neuron_type,
                                           surrogate_function=args.surrogate_function, neuron_args={},
                                           is_video=True, ).to(device)
    _logger.info(
        f"Model created: {args.model} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    loader_train, loader_eval, loader_test = create_dataloaders(args, _logger)
    mixup_fn = create_mixup_fn(args, _logger)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, num_epochs = create_scheduler(args, optimizer)

    criterion_train_soft = SoftTargetCrossEntropy().to(device)
    criterion_train_hard = (LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(
        device) if args.smoothing > 0 else nn.CrossEntropyLoss().to(device))
    validate_loss_fn = nn.CrossEntropyLoss().to(device)

    loss_scaler = NativeScaler() if args.amp else None
    start_epoch = args.start_epoch
    best_acc = 0.0
    patience_counter = 0  # Initialize patience counter for early stopping

    _logger.info(f"Scheduled to run for {num_epochs} epochs.")
    _logger.info(f"Early stopping is enabled with a patience of {args.patience_epochs} epochs.")
    _logger.info(f"Plots will be updated every {args.plot_update_interval} epochs.")
    _logger.info("Starting training...")

    total_start_time = time.time()  # Start total timer

    for epoch in range(start_epoch, num_epochs):
        train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, criterion_train_soft,
                                        criterion_train_hard, args, lr_scheduler=scheduler, loss_scaler=loss_scaler,
                                        mixup_fn=mixup_fn, )
        eval_metrics = evaluate(model, loader_eval, validate_loss_fn, args)

        if scheduler:
            scheduler.step(epoch + 1, eval_metrics["top1"])

        val_acc = eval_metrics["top1"]
        is_best = val_acc > best_acc

        # Store metrics for plotting
        train_losses.append(train_metrics["loss"])
        train_accuracies.append(train_metrics["top1"])
        val_losses.append(eval_metrics["loss"])
        val_accuracies.append(eval_metrics["top1"])

        # Modified Epoch Summary Logging
        lr = optimizer.param_groups[0]["lr"]

        if is_best:
            best_acc = val_acc
            patience_counter = 0  # Reset counter on improvement
            best_checkpoint_path = os.path.join(output_dir, "best.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "best_acc": best_acc, },
                       best_checkpoint_path, )
            _logger.info(f"*** Saved new best model with Acc@1: {best_acc:.2f}% at epoch {epoch}")
        else:
            patience_counter += 1
            _logger.info(f"No improvement in validation accuracy for {patience_counter} epoch(s).")

        _logger.info(
            f"Epoch {epoch} Summary | Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['top1']:.2f}% | "
            f"Val Loss: {eval_metrics['loss']:.4f} | Val Acc@1: {val_acc:.2f}% | "
            f"Train Time: {train_metrics['duration']:.2f}s | Avg Data Time: {train_metrics['data_time']:.3f}s | "
            f"LR: {lr:.6f} | Best Acc: {best_acc:.2f}%")

        # Update plots at specified intervals
        if (epoch + 1) % args.plot_update_interval == 0:
            _logger.info(f"Updating plots at epoch {epoch + 1}...")
            plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, output_dir, _logger)

        # Check for early stopping
        if args.patience_epochs > 0 and patience_counter >= args.patience_epochs:
            _logger.info(
                f"Early stopping triggered after {args.patience_epochs} epochs of no improvement. Best validation accuracy: {best_acc:.2f}%")
            break  # Exit the training loop

    total_duration_seconds = time.time() - total_start_time
    total_duration_str = str(datetime.timedelta(seconds=int(total_duration_seconds)))
    _logger.info(f"Training finished. Total duration: {total_duration_str}. Best validation accuracy: {best_acc:.2f}%")

    # Load best model and evaluate on test set
    _logger.info("Loading best model for final test evaluation...")
    best_checkpoint_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint["model"])
        _logger.info(
            f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['best_acc']:.2f}%")

        # Evaluate on test set
        test_metrics = test_final_model(model, loader_test, validate_loss_fn, args, _logger)

        # Save test results
        test_results_path = os.path.join(output_dir, "test_results.txt")
        with open(test_results_path, "w") as f:
            f.write(f"Final Test Results:\n")
            f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"Test Accuracy (Top-1): {test_metrics['top1']:.2f}%\n")
            f.write(f"Test Accuracy (Top-5): {test_metrics['top5']:.2f}%\n")
            f.write(f"Test Duration: {test_metrics['duration']:.2f}s\n")
            f.write(f"Best Validation Accuracy: {best_acc:.2f}%\n")
        _logger.info(f"Test results saved to: {test_results_path}")
    else:
        _logger.warning("Best model checkpoint not found. Skipping test evaluation.")

    # Final plot update
    _logger.info("Generating final plots...")
    plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, output_dir, _logger)


def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, output_dir, _logger):
    """Plots accuracy and loss curves with line width 2 and shows peak/lowest values."""
    epochs_range = range(len(train_accuracies))

    # Create single figure with clean styling for accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Set clean background and grid
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    # Plot training and validation accuracy with line width 2
    ax.plot(epochs_range, train_accuracies, label="Training Accuracy", linewidth=2, color="#1f77b4")
    ax.plot(epochs_range, val_accuracies, label="Validation Accuracy", linewidth=2, color="#17becf")

    # Find and mark peak accuracies
    if train_accuracies:
        peak_train_acc = max(train_accuracies)
        peak_train_epoch = train_accuracies.index(peak_train_acc)
        ax.plot(peak_train_epoch, peak_train_acc, 'o', color="#1f77b4", markersize=8,
                label=f'Peak Train Acc: {peak_train_acc:.2f}%')

    if val_accuracies:
        peak_val_acc = max(val_accuracies)
        peak_val_epoch = val_accuracies.index(peak_val_acc)
        ax.plot(peak_val_epoch, peak_val_acc, 'o', color="#17becf", markersize=8,
                label=f'Peak Val Acc: {peak_val_acc:.2f}%')

    # Set labels and title
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.tick_params(labelsize=12)

    # Position legend in upper left
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Set clean spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('gray')

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    _logger.info(f"Accuracy plot saved to: {plot_path}")

    # Create loss plot with line width 2 and show lowest values
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(10, 6))
    ax_loss.set_facecolor('white')
    ax_loss.grid(True, alpha=0.3, linewidth=0.8)
    ax_loss.set_axisbelow(True)

    ax_loss.plot(epochs_range, train_losses, label="Training Loss", linewidth=2, color="#1f77b4")
    ax_loss.plot(epochs_range, val_losses, label="Validation Loss", linewidth=2, color="#17becf")

    # Find and mark lowest losses
    if train_losses:
        lowest_train_loss = min(train_losses)
        lowest_train_epoch = train_losses.index(lowest_train_loss)
        ax_loss.plot(lowest_train_epoch, lowest_train_loss, 'o', color="#1f77b4", markersize=8,
                     label=f'Lowest Train Loss: {lowest_train_loss:.4f}')

    if val_losses:
        lowest_val_loss = min(val_losses)
        lowest_val_epoch = val_losses.index(lowest_val_loss)
        ax_loss.plot(lowest_val_epoch, lowest_val_loss, 'o', color="#17becf", markersize=8,
                     label=f'Lowest Val Loss: {lowest_val_loss:.4f}')

    ax_loss.set_xlabel('Epoch', fontsize=14)
    ax_loss.set_ylabel('Loss', fontsize=14)
    ax_loss.tick_params(labelsize=12)
    ax_loss.legend(loc='upper right', fontsize=11, framealpha=0.9)

    for spine in ax_loss.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('gray')

    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    _logger.info(f"Loss plot saved to: {loss_plot_path}")


if __name__ == "__main__":
    main()
