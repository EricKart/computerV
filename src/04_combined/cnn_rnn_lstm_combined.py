"""
============================================================
  MODULE 4 — COMBINED CNN + RNN + LSTM MODEL
  ============================================
  Architecture:  CNN Feature Extractor  +  LSTM Temporal Encoder
  Dataset:       CIFAR-10  (treated as simulated video frames)
  Framework:     PyTorch

  THE BIG IDEA
  ────────────
  Real-world video/sequence classification uses:
    CNN  →  extract spatial features from each frame/patch
    LSTM →  model temporal relationships across frames

  Since CIFAR-10 has single images (no video), we *simulate*
  a short video by splitting each 32×32 image into a sequence
  of overlapping 16×16 patches:

    ┌────────┬────────┐      Patch extraction (stride=8):
    │ Patch1 │ Patch2 │        → 9 patches of 16×16
    │        │        │        → CNN encodes each to 128-dim
    ├────────┼────────┤        → LSTM reads the 9-step sequence
    │ Patch3 │ Patch4 │        → Final hidden → classify
    │        │        │
    └────────┴────────┘

  This mirrors real architectures like:
    • CNN + LSTM for video classification
    • CNN + LSTM for action recognition
    • CNN + LSTM for image captioning

  RUN:
    cd <project_root>
    python -m src.04_combined.cnn_rnn_lstm_combined
============================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
#  HYPER-PARAMETERS
# ============================================================
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 20
NUM_CLASSES   = 10

# Patch extraction
PATCH_SIZE    = 16        # each patch is 16×16 pixels
PATCH_STRIDE  = 8         # overlap of 8 pixels between patches
# For a 32×32 image with patch=16, stride=8:
#   positions along each axis: (32 - 16) / 8 + 1 = 3
#   total patches: 3 × 3 = 9

# CNN feature extractor
CNN_FEATURES  = 128       # output dim from the CNN backbone

# LSTM
LSTM_HIDDEN   = 256
LSTM_LAYERS   = 2


# ============================================================
#  CNN BACKBONE  (feature extractor for single patches)
# ============================================================
class PatchCNN(nn.Module):
    """
    A small CNN that converts a 16×16×3 patch into a 128-dim
    feature vector.

    Flow:
      (3, 16, 16)
        → Conv(3→32, 3×3) + BN + ReLU + Pool(2)  → (32, 8, 8)
        → Conv(32→64, 3×3) + BN + ReLU + Pool(2) → (64, 4, 4)
        → Flatten → (1024)
        → Linear(1024 → 128)
        → (128)    ← one feature vector per patch
    """

    def __init__(self, out_features: int = CNN_FEATURES):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ──
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,16,16)→(32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (32,8,8)

            # ── Block 2 ──
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # → (64,4,4)
        )

        # 64 channels × 4 × 4 spatial = 1024
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: (batch, 3, 16, 16) → (batch, 128)"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
#  COMBINED CNN + LSTM MODEL
# ============================================================
class CombinedCNNLSTM(nn.Module):
    """
    Full architecture combining CNN and LSTM.

    Pipeline
    ════════

    Input Image (3, 32, 32)
         │
         ▼
    ┌──────────────────────────────────────────┐
    │  PATCH EXTRACTION                         │
    │  Split into 9 overlapping 16×16 patches   │
    │  (3×3 grid with stride=8)                 │
    └──────────────────────────────────────────┘
         │
         ▼  9 patches, each (3, 16, 16)
    ┌──────────────────────────────────────────┐
    │  CNN BACKBONE  (PatchCNN)                 │
    │  Each patch → 128-dim feature vector      │
    │  Applied identically to all 9 patches     │
    │  (weight sharing = same CNN for each)      │
    └──────────────────────────────────────────┘
         │
         ▼  sequence of 9 vectors, each 128-dim
    ┌──────────────────────────────────────────┐
    │  BIDIRECTIONAL LSTM  (2 layers)           │
    │  Reads the 9-step feature sequence        │
    │  Forward + Backward = 512-dim output      │
    │  Models spatial relationships between      │
    │  patches (analogous to temporal in video)  │
    └──────────────────────────────────────────┘
         │
         ▼  concatenated final hidden: 512
    ┌──────────────────────────────────────────┐
    │  ATTENTION POOLING                        │
    │  Weighted average of all 9 LSTM outputs   │
    │  Lets the model focus on important patches │
    └──────────────────────────────────────────┘
         │
         ▼  512-dim context vector
    ┌──────────────────────────────────────────┐
    │  CLASSIFIER HEAD                          │
    │  LayerNorm → Dropout → FC(512→256)       │
    │  → ReLU → Dropout → FC(256→10)           │
    └──────────────────────────────────────────┘
         │
         ▼
    10 class logits
    """

    def __init__(
        self,
        cnn_features:  int = CNN_FEATURES,
        lstm_hidden:   int = LSTM_HIDDEN,
        lstm_layers:   int = LSTM_LAYERS,
        num_classes:   int = NUM_CLASSES,
    ):
        super().__init__()

        self.patch_size   = PATCH_SIZE
        self.patch_stride = PATCH_STRIDE

        # ── CNN that processes each patch ─────────────────
        self.cnn = PatchCNN(out_features=cnn_features)

        # ── LSTM that reads the patch sequence ────────────
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        lstm_output_dim = lstm_hidden * 2  # bidirectional

        # ── Simple attention mechanism ────────────────────
        # Learn a single attention vector that scores each
        # time step.  Weighted sum → context vector.
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 1),
        )

        # ── Classifier ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(0.5),
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def extract_patches(self, images):
        """
        Extract a grid of overlapping patches from each image.

        Parameters
        ----------
        images : (batch, 3, 32, 32)

        Returns
        -------
        patches : (batch, num_patches, 3, patch_size, patch_size)

        For 32×32 images with patch=16, stride=8:
          positions per axis:  [0, 8, 16]  → 3 positions
          total patches: 3 × 3 = 9
        """
        # Use unfold to extract sliding windows along H and W
        # unfold(dim, size, step)
        patches = images.unfold(2, self.patch_size, self.patch_stride)  # along H
        patches = patches.unfold(3, self.patch_size, self.patch_stride) # along W
        # Shape now: (batch, 3, n_h, n_w, patch_size, patch_size)

        batch, C, n_h, n_w, ph, pw = patches.shape
        num_patches = n_h * n_w

        # Rearrange to (batch, num_patches, C, ph, pw)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(batch, num_patches, C, ph, pw)

        return patches

    def forward(self, x):
        """
        x : (batch, 3, 32, 32) → logits : (batch, 10)
        """
        batch_size = x.size(0)

        # ── Step 1: Extract patches ───────────────────────
        patches = self.extract_patches(x)   # (B, 9, 3, 16, 16)
        num_patches = patches.size(1)

        # ── Step 2: CNN encodes each patch ────────────────
        # Flatten batch and patch dims so CNN sees (B*9, 3, 16, 16)
        patches_flat = patches.reshape(-1, 3, self.patch_size, self.patch_size)
        cnn_features = self.cnn(patches_flat)  # (B*9, 128)

        # Reshape back to sequence: (B, 9, 128)
        cnn_features = cnn_features.reshape(batch_size, num_patches, -1)

        # ── Step 3: LSTM processes the sequence ───────────
        lstm_out, _ = self.lstm(cnn_features)  # (B, 9, 512)

        # ── Step 4: Attention pooling ─────────────────────
        # Score each time step
        attn_scores = self.attention(lstm_out)      # (B, 9, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # normalise over seq

        # Weighted sum → context vector
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, 512)

        # ── Step 5: Classify ──────────────────────────────
        logits = self.classifier(context)   # (B, 10)
        return logits


# ============================================================
#  TRAINING & EVALUATION
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

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

    return running_loss / total, 100.0 * correct / total, all_true, all_pred


# ============================================================
#  MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  MODULE 4: COMBINED CNN + LSTM MODEL")
    print("  Dataset : CIFAR-10  |  Model : PatchCNN + BiLSTM")
    print("  Pipeline: Image → Patches → CNN → LSTM → Classify")
    print("=" * 60)

    device = get_device()
    train_loader, test_loader, class_names = get_cifar10_loaders(
        batch_size=BATCH_SIZE, augment=True,
    )

    model = CombinedCNNLSTM().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Patches per image: {3 * 3} = 9  (16×16 with stride 8)")
    print(f"CNN output dim:    {CNN_FEATURES}")
    print(f"LSTM hidden:       {LSTM_HIDDEN} × 2 (bidirectional)\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []
    best_acc = 0.0

    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        te_loss, te_acc, _, _ = evaluate(
            model, test_loader, criterion, device,
        )
        scheduler.step()
        elapsed = time.time() - start

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

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "combined_best.pth"))

    print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")
    print("ℹ️  The combined model leverages CNN's spatial features")
    print("   with LSTM's sequential modelling for best results.\n")

    # ── Final evaluation ──
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "combined_best.pth"), weights_only=True,
    ))
    _, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device,
    )
    print(f"📋 Final Test Accuracy (best checkpoint): {final_acc:.2f}%\n")

    # ── Visualisations ──
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        title="Combined CNN+LSTM Training Curves",
        filename="combined_training_curves.png",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="Combined CNN+LSTM Confusion Matrix",
        filename="combined_confusion_matrix.png",
    )
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        title="Combined CNN+LSTM Per-Class Accuracy",
        filename="combined_per_class_accuracy.png",
    )

    sample_images, sample_labels = next(iter(test_loader))
    sample_outputs = model(sample_images.to(device))
    _, sample_preds = sample_outputs.max(1)

    plot_sample_predictions(
        denormalize_cifar10(sample_images),
        sample_labels.tolist(),
        sample_preds.cpu().tolist(),
        class_names,
        n=16,
        title="Combined CNN+LSTM Sample Predictions",
        filename="combined_sample_predictions.png",
    )

    # ── Save ──
    final_path = os.path.join(model_dir, "combined_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved → {final_path}")

    # ── ONNX export ──
    try:
        dummy = torch.randn(1, 3, 32, 32, device=device)
        onnx_path = os.path.join(model_dir, "combined_model.onnx")
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"📦 ONNX model exported → {onnx_path}")
    except Exception as e:
        print(f"⚠️  ONNX export skipped: {e}")

    # ── Print model comparison ──
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print("  ┌──────────────────┬────────────────────────────┐")
    print("  │ Model            │ Expected Accuracy (CIFAR-10)│")
    print("  ├──────────────────┼────────────────────────────┤")
    print("  │ CNN (Module 1)   │ ~80-85%                    │")
    print("  │ RNN (Module 2)   │ ~50-55%                    │")
    print("  │ LSTM (Module 3)  │ ~60-65%                    │")
    print("  │ Combined (Mod 4) │ ~75-82%                    │")
    print("  └──────────────────┴────────────────────────────┘")
    print()
    print("  KEY TAKEAWAYS:")
    print("  • CNN excels at spatial features (best for single images)")
    print("  • RNN/LSTM alone are weaker on images (designed for sequences)")
    print("  • Combined CNN+LSTM shines on video/sequential visual data")
    print("  • In real video tasks, Combined often beats pure CNN")
    print()
    print("✅ Combined model training complete!  Check outputs/ for plots.")


if __name__ == "__main__":
    main()
