"""
============================================================
  MODULE 2 — RNN SEQUENCE MODEL
  ==============================
  Architecture:  2-layer Vanilla RNN  +  Fully-Connected Head
  Dataset:       CIFAR-10  (32×32 RGB, 10 classes)
  Framework:     PyTorch

  KEY IDEA — IMAGES AS SEQUENCES
  ───────────────────────────────
  A 32×32×3 image is reshaped into a *sequence* of 32 time steps,
  where each step is one row of the image (32 pixels × 3 channels
  = 96 features).  The RNN reads the image row-by-row from top to
  bottom, building up context in its hidden state.

        Row  0  ──▶ RNN Cell ──▶ h₁
        Row  1  ──▶ RNN Cell ──▶ h₂
        Row  2  ──▶ RNN Cell ──▶ h₃
          ⋮              ⋮         ⋮
        Row 31  ──▶ RNN Cell ──▶ h₃₂  ──▶  FC  ──▶  10 logits

  WHY THIS MATTERS
  ────────────────
  • Demonstrates sequence modelling on real image data
  • Shows the vanishing-gradient limitation of vanilla RNNs
  • Sets the stage for the LSTM improvement in Module 3

  RUN:
    cd <project_root>
    python -m src.02_rnn.rnn_sequence_model
============================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- Fix imports ----------
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
EPOCHS        = 15
NUM_CLASSES   = 10

# RNN-specific
SEQUENCE_LEN  = 32       # number of rows in a 32×32 image
INPUT_SIZE    = 96        # 32 pixels × 3 channels per row
HIDDEN_SIZE   = 256       # dimensionality of the hidden state
NUM_LAYERS    = 2         # stacked RNN layers


# ============================================================
#  RNN MODEL DEFINITION
# ============================================================
class CifarRNN(nn.Module):
    """
    Vanilla RNN treating each 32×32×3 image as a sequence of
    32 row-vectors (each 96-dimensional).

    Architecture
    ═══════════
    Image (3, 32, 32)
      │
      ▼  reshape → (32, 96)   ← 32 time steps, 96 features each
    ┌───────────────────────────────┐
    │ RNN Layer 1                   │
    │   h_t = tanh(W_ih · x_t      │  ← standard RNN formula
    │          + W_hh · h_{t-1}     │
    │          + bias)              │
    │   hidden_size = 256           │
    ├───────────────────────────────┤
    │ RNN Layer 2  (stacked)        │
    │   reads h_t from layer 1     │
    │   produces deeper features    │
    │   hidden_size = 256           │
    └───────────────────────────────┘
      │
      ▼  take h₃₂ (last time step)
    ┌───────────────────────────────┐
    │ Fully-Connected Head          │
    │   LayerNorm(256)              │
    │   Linear(256 → 10)           │
    └───────────────────────────────┘
      │
      ▼
    10 logits
    """

    def __init__(
        self,
        input_size:  int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ────────────────────────────────────────────────────
        # nn.RNN Parameters:
        #   input_size  – features per time step (96)
        #   hidden_size – size of the hidden state  (256)
        #   num_layers  – number of stacked RNN layers (2)
        #   batch_first – input shape is (batch, seq, features)
        #   dropout     – dropout between stacked layers
        # ────────────────────────────────────────────────────
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq_len, input_size)
            dropout=0.3,              # dropout between layers 1 & 2
            nonlinearity="tanh",       # standard tanh activation
        )

        # ── Classification head ──
        # LayerNorm stabilises the final hidden state before the FC
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : tensor of shape (batch, 3, 32, 32)

        Returns
        -------
        logits : tensor of shape (batch, num_classes)
        """

        batch_size = x.size(0)

        # ── STEP 1: Reshape image into a sequence ─────────
        #   (batch, 3, 32, 32) → (batch, 32, 32, 3)  → (batch, 32, 96)
        #   Each of the 32 rows becomes a 96-dim feature vector.
        x = x.permute(0, 2, 3, 1)             # channels last
        x = x.reshape(batch_size, SEQUENCE_LEN, INPUT_SIZE)

        # ── STEP 2: Initialise hidden state with zeros ────
        #   Shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=x.device,
        )

        # ── STEP 3: Run the RNN over all 32 time steps ───
        #   out  : (batch, 32, 256)  — hidden state at every step
        #   h_n  : (2, batch, 256)   — final hidden state per layer
        out, h_n = self.rnn(x, h0)

        # ── STEP 4: Take the LAST time step's output ─────
        #   We only care about the hidden state after seeing
        #   the entire image (all 32 rows).
        last_hidden = out[:, -1, :]            # (batch, 256)

        # ── STEP 5: Classify ─────────────────────────────
        logits = self.classifier(last_hidden)  # (batch, 10)
        return logits


# ============================================================
#  TRAINING LOOP  (one epoch)
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # ── Gradient clipping ─────────────────────────────
        # RNNs are prone to exploding gradients during BPTT.
        # Clipping the gradient norm to 5 prevents wild updates.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


# ============================================================
#  EVALUATION LOOP
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
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
    print("  MODULE 2: RNN SEQUENCE MODEL")
    print("  Dataset : CIFAR-10  |  Model : CifarRNN")
    print("  Approach: Image → 32 row-vectors → RNN → classify")
    print("=" * 60)

    device = get_device()
    train_loader, test_loader, class_names = get_cifar10_loaders(
        batch_size=BATCH_SIZE, augment=True,
    )

    model = CifarRNN().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, 
    )

    # ── Training ──
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
        scheduler.step(te_loss)
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
            torch.save(model.state_dict(), os.path.join(model_dir, "rnn_best.pth"))

    print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")
    print("ℹ️  Note: Vanilla RNNs typically reach 50-60% on CIFAR-10.")
    print("   The LSTM in Module 3 will do better!\n")

    # ── Final evaluation ──
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "rnn_best.pth"), weights_only=True,
    ))
    _, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device,
    )

    # ── Visualisations ──
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        title="RNN Training Curves", filename="rnn_training_curves.png",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="RNN Confusion Matrix", filename="rnn_confusion_matrix.png",
    )
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        title="RNN Per-Class Accuracy", filename="rnn_per_class_accuracy.png",
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
        title="RNN Sample Predictions",
        filename="rnn_sample_predictions.png",
    )

    # ── Save ──
    final_path = os.path.join(model_dir, "rnn_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved → {final_path}")
    print("\n✅ RNN training complete!  Check outputs/ for plots.")


if __name__ == "__main__":
    main()
