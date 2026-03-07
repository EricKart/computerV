"""
============================================================
  MODULE 1 — CNN IMAGE CLASSIFIER
  ================================
  Architecture:  3 Convolutional Blocks  +  2 Fully-Connected Layers
  Dataset:       CIFAR-10  (32×32 RGB, 10 classes)
  Framework:     PyTorch

  WHAT THIS SCRIPT DOES (step by step):
    1. Load and augment CIFAR-10 data
    2. Define a CNN model from scratch
    3. Train for a configurable number of epochs
    4. Evaluate on the test set
    5. Save the trained model & produce visualisations

  RUN:
    cd <project_root>
    python -m src.01_cnn.cnn_image_classifier
============================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------- Fix imports when running as a module ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_loader import get_cifar10_loaders, get_device
from src.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_per_class_accuracy,
    denormalize_cifar10,
)


# ============================================================
#  HYPER-PARAMETERS  (tweak these to experiment)
# ============================================================
BATCH_SIZE    = 64       # images per gradient update
LEARNING_RATE = 0.001    # Adam step size
EPOCHS        = 15       # full passes over the training set
NUM_CLASSES   = 10       # CIFAR-10 has 10 categories


# ============================================================
#  CNN MODEL DEFINITION
# ============================================================
class CifarCNN(nn.Module):
    """
    A simple but effective CNN for 32×32 colour images.

    Architecture Diagram
    ═══════════════════
    Input (3×32×32)
      │
      ▼
    ┌─────────────────────────────────────┐
    │ CONV BLOCK 1                        │
    │  Conv2d(3→32, 3×3, pad=1)          │  ← 32 filters detect edges
    │  BatchNorm2d(32)                    │  ← stabilise training
    │  ReLU                               │  ← introduce non-linearity
    │  Conv2d(32→32, 3×3, pad=1)         │  ← deeper edges / textures
    │  BatchNorm2d(32)                    │
    │  ReLU                               │
    │  MaxPool2d(2×2)                    │  ← downsample 32→16
    │  Dropout(0.25)                      │  ← regularise
    └─────────────────────────────────────┘
      │  Output: 32 × 16 × 16
      ▼
    ┌─────────────────────────────────────┐
    │ CONV BLOCK 2                        │
    │  Conv2d(32→64, 3×3, pad=1)         │  ← 64 filters detect shapes
    │  BatchNorm2d(64)                    │
    │  ReLU                               │
    │  Conv2d(64→64, 3×3, pad=1)         │
    │  BatchNorm2d(64)                    │
    │  ReLU                               │
    │  MaxPool2d(2×2)                    │  ← downsample 16→8
    │  Dropout(0.25)                      │
    └─────────────────────────────────────┘
      │  Output: 64 × 8 × 8
      ▼
    ┌─────────────────────────────────────┐
    │ CONV BLOCK 3                        │
    │  Conv2d(64→128, 3×3, pad=1)        │  ← 128 filters detect objects
    │  BatchNorm2d(128)                   │
    │  ReLU                               │
    │  Conv2d(128→128, 3×3, pad=1)       │
    │  BatchNorm2d(128)                   │
    │  ReLU                               │
    │  MaxPool2d(2×2)                    │  ← downsample 8→4
    │  Dropout(0.25)                      │
    └─────────────────────────────────────┘
      │  Output: 128 × 4 × 4  →  flatten → 2048
      ▼
    ┌─────────────────────────────────────┐
    │ CLASSIFIER HEAD                     │
    │  Linear(2048 → 512)                │
    │  BatchNorm1d(512)                   │
    │  ReLU                               │
    │  Dropout(0.5)                       │
    │  Linear(512 → 10)                  │  ← one score per class
    └─────────────────────────────────────┘
      │
      ▼
    Output: 10 logits  (pass to CrossEntropyLoss)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Conv Block 1 ──────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,32,32) → (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (32,32,32) → (32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (32,32,32) → (32,16,16)
            nn.Dropout2d(0.25),
        )

        # ── Conv Block 2 ──────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32,16,16) → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # (64,16,16) → (64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (64,16,16) → (64,8,8)
            nn.Dropout2d(0.25),
        )

        # ── Conv Block 3 ──────────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (64,8,8) → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# (128,8,8) → (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (128,8,8) → (128,4,4)
            nn.Dropout2d(0.25),
        )

        # ── Fully-Connected Classifier ────────────────────────
        # After block3 the feature map is 128 × 4 × 4 = 2048 values
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Forward pass:
          x shape: (batch, 3, 32, 32)
          returns:  (batch, num_classes)  — raw logits (no softmax)
        """
        x = self.block1(x)        # → (batch, 32, 16, 16)
        x = self.block2(x)        # → (batch, 64,  8,  8)
        x = self.block3(x)        # → (batch, 128, 4,  4)

        # Flatten all spatial dims into one vector per image
        x = x.view(x.size(0), -1) # → (batch, 2048)

        x = self.classifier(x)    # → (batch, 10)
        return x


# ============================================================
#  TRAINING LOOP  (one epoch)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train for one full pass over `loader`.

    Returns
    -------
    (average_loss, accuracy_percent)
    """
    model.train()                  # enable dropout & batch-norm training mode
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        # 1. Move data to GPU (or CPU)
        images, labels = images.to(device), labels.to(device)

        # 2. Forward pass — compute predictions
        outputs = model(images)

        # 3. Compute loss (CrossEntropy combines LogSoftmax + NLLLoss)
        loss = criterion(outputs, labels)

        # 4. Backward pass — compute gradients
        optimizer.zero_grad()      # clear old gradients
        loss.backward()            # compute new gradients

        # 5. Update weights
        optimizer.step()

        # ── Track statistics ──
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)          # class with highest score
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============================================================
#  EVALUATION LOOP
# ============================================================
@torch.no_grad()   # disable gradient computation → saves memory & time
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on `loader` (no gradient updates).

    Returns
    -------
    (average_loss, accuracy_percent, all_true_labels, all_pred_labels)
    """
    model.eval()                   # disable dropout & batch-norm updates
    running_loss = 0.0
    correct = 0
    total   = 0
    all_true = []
    all_pred = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(predicted.cpu().tolist())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, all_true, all_pred


# ============================================================
#  MAIN — ORCHESTRATE EVERYTHING
# ============================================================
def main():
    print("=" * 60)
    print("  MODULE 1: CNN IMAGE CLASSIFIER")
    print("  Dataset : CIFAR-10  |  Model : CifarCNN")
    print("=" * 60)

    # ---- 1. Setup ----
    device = get_device()
    train_loader, test_loader, class_names = get_cifar10_loaders(
        batch_size=BATCH_SIZE, augment=True,
    )

    # ---- 2. Instantiate model, loss, optimiser ----
    model = CifarCNN(num_classes=NUM_CLASSES).to(device)

    # Print a summary of the model's layers and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {train_params:,}\n")

    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimiser — adaptive learning rate, works well out of the box
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Reduce LR by 0.1× when test loss plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer, mode="min", factor=0.1, patience=3
        optimizer, mode="min", factor=0.1, patience=3,

    )

    # ---- 3. Training loop ----
    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []
    best_acc = 0.0

    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        # Evaluate
        te_loss, te_acc, _, _ = evaluate(
            model, test_loader, criterion, device,
        )

        # Step the LR scheduler based on test loss
        scheduler.step(te_loss)

        elapsed = time.time() - start

        # Record history for plotting
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}]  "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}%  │  "
            f"Test  Loss: {te_loss:.4f}  Acc: {te_acc:.2f}%  │  "
            f"{elapsed:.1f}s"
        )

        # Save best model checkpoint
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "cnn_best.pth"))

    print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")

    # ---- 4. Final evaluation & visualisations ----
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "cnn_best.pth"), weights_only=True,
    ))
    _, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device,
    )
    print(f"📋 Final Test Accuracy (best checkpoint): {final_acc:.2f}%\n")

    # Training curves
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        title="CNN Training Curves", filename="cnn_training_curves.png",
    )

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="CNN Confusion Matrix", filename="cnn_confusion_matrix.png",
    )

    # Per-class accuracy
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        title="CNN Per-Class Accuracy", filename="cnn_per_class_accuracy.png",
    )

    # Sample predictions (first 16 test images)
    sample_images, sample_labels = next(iter(test_loader))
    sample_outputs = model(sample_images.to(device))
    _, sample_preds = sample_outputs.max(1)

    plot_sample_predictions(
        denormalize_cifar10(sample_images),
        sample_labels.tolist(),
        sample_preds.cpu().tolist(),
        class_names,
        n=16,
        title="CNN Sample Predictions",
        filename="cnn_sample_predictions.png",
    )

    # ---- 5. Save final model ----
    final_path = os.path.join(model_dir, "cnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Final model saved → {final_path}")

    # ---- 6. Export to ONNX (for Azure deployment) ----
    try:
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        onnx_path   = os.path.join(model_dir, "cnn_model.onnx")
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"📦 ONNX model exported → {onnx_path}")
    except Exception as e:
        print(f"⚠️  ONNX export skipped: {e}")

    print("\n✅ CNN training complete!  Check the outputs/ folder for plots.")


# ============================================================
if __name__ == "__main__":
    main()
