"""
============================================================
  MODULE 3 — LSTM MODEL
  ======================
  Architecture:  2-layer LSTM  +  Fully-Connected Head
  Dataset:       CIFAR-10  (32×32 RGB, 10 classes)
  Framework:     PyTorch

  WHY LSTM OVER RNN?
  ──────────────────
  The vanilla RNN in Module 2 suffers from the vanishing-gradient
  problem: gradients shrink exponentially during BPTT, making it
  hard to capture long-range dependencies.

  The LSTM fixes this with a gated memory cell:
    • Forget gate  (fₜ) — what to erase from memory
    • Input gate   (iₜ) — what new info to write
    • Output gate  (oₜ) — what to expose as hidden state

  These gates let gradients flow through the cell state with
  minimal decay, enabling the network to "remember" information
  from early rows of the image when it reaches the last row.

  SAME INPUT FORMAT AS MODULE 2
  ─────────────────────────────
  Image (3, 32, 32)  →  sequence of 32 rows  →  LSTM  →  classify

  EXPECTED IMPROVEMENT
  ────────────────────
  LSTM should reach ~60-65% on CIFAR-10 (vs ~50-55% for vanilla RNN)
  because it can retain spatial patterns across distant rows.

  RUN:
    cd <project_root>
    python -m src.03_lstm.lstm_model
============================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

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

SEQUENCE_LEN  = 32        # rows per image
INPUT_SIZE    = 96         # 32 px × 3 channels per row
HIDDEN_SIZE   = 256        # LSTM hidden dimension
NUM_LAYERS    = 2          # stacked LSTM layers
BIDIRECTIONAL = True       # read image top-to-bottom AND bottom-to-top


# ============================================================
#  LSTM MODEL DEFINITION
# ============================================================
class CifarLSTM(nn.Module):
    """
    Bidirectional LSTM for CIFAR-10 image classification.

    Architecture
    ═══════════
    Image (3, 32, 32)
      │
      ▼  reshape → (32, 96)
    ┌───────────────────────────────────────────────┐
    │ Bidirectional LSTM Layer 1                     │
    │                                               │
    │  Forward:  reads rows 0 → 31  (top to bottom) │
    │  Backward: reads rows 31 → 0  (bottom to top) │
    │                                               │
    │  Each direction has hidden_size = 256          │
    │  Combined output per step: 512                 │
    ├───────────────────────────────────────────────┤
    │ Bidirectional LSTM Layer 2  (stacked)          │
    │  Input: 512 from layer 1                       │
    │  Output per step: 512                          │
    └───────────────────────────────────────────────┘
      │
      ▼  concat forward h₃₂ + backward h₀ → 512
    ┌───────────────────────────────────────────────┐
    │ Classifier Head                               │
    │  LayerNorm(512)                               │
    │  Dropout(0.5)                                 │
    │  Linear(512 → 128)                           │
    │  ReLU                                         │
    │  Dropout(0.3)                                 │
    │  Linear(128 → 10)                            │
    └───────────────────────────────────────────────┘
      │
      ▼
    10 logits

    WHY BIDIRECTIONAL?
    ──────────────────
    A unidirectional LSTM reading top-to-bottom only sees "above"
    when classifying.  A bidirectional LSTM also sees "below",
    giving it full context of the image — like reading a sentence
    forwards *and* backwards.
    """

    def __init__(
        self,
        input_size:    int = INPUT_SIZE,
        hidden_size:   int = HIDDEN_SIZE,
        num_layers:    int = NUM_LAYERS,
        num_classes:   int = NUM_CLASSES,
        bidirectional: bool = BIDIRECTIONAL,
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_dirs      = 2 if bidirectional else 1

        # ────────────────────────────────────────────────────
        # nn.LSTM internals (per layer, per direction):
        #
        #   fₜ = σ(W_if · xₜ + W_hf · h_{t-1} + b_f)    ← Forget gate
        #   iₜ = σ(W_ii · xₜ + W_hi · h_{t-1} + b_i)    ← Input gate
        #   gₜ = tanh(W_ig · xₜ + W_hg · h_{t-1} + b_g) ← Candidate cell
        #   oₜ = σ(W_io · xₜ + W_ho · h_{t-1} + b_o)    ← Output gate
        #
        #   cₜ = fₜ ⊙ c_{t-1}  +  iₜ ⊙ gₜ              ← Cell state update
        #   hₜ = oₜ ⊙ tanh(cₜ)                           ← Hidden state
        # ────────────────────────────────────────────────────

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional,
        )

        # The classifier receives the concatenated final hidden
        # states from both directions: hidden_size × num_dirs
        classifier_input = hidden_size * self.num_dirs

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(0.5),
            nn.Linear(classifier_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x : (batch, 3, 32, 32) → logits : (batch, 10)
        """
        batch_size = x.size(0)

        # ── Reshape to sequence ───────────────────────────
        x = x.permute(0, 2, 3, 1)                       # (B, 32, 32, 3)
        x = x.reshape(batch_size, SEQUENCE_LEN, INPUT_SIZE)  # (B, 32, 96)

        # ── Initialise hidden state (h₀) and cell state (c₀) ─
        #   Shape: (num_layers × num_dirs, batch, hidden_size)
        h0 = torch.zeros(
            self.num_layers * self.num_dirs, batch_size,
            self.hidden_size, device=x.device,
        )
        c0 = torch.zeros_like(h0)

        # ── Run LSTM ──────────────────────────────────────
        # out shape: (batch, 32, hidden_size × num_dirs)
        # h_n shape: (num_layers × num_dirs, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # ── Extract final hidden states ───────────────────
        if self.bidirectional:
            # Forward direction:  h_n[-2]  (last layer, forward)
            # Backward direction: h_n[-1]  (last layer, backward)
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 512)
        else:
            last_hidden = h_n[-1]                               # (B, 256)

        logits = self.classifier(last_hidden)
        return logits


# ============================================================
#  TRAINING & EVALUATION  (same pattern as Module 2)
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
    print("  MODULE 3: LSTM MODEL")
    print("  Dataset : CIFAR-10  |  Model : Bidirectional LSTM")
    print("  Approach: Image → 32 row-vectors → BiLSTM → classify")
    print("=" * 60)

    device = get_device()
    train_loader, test_loader, class_names = get_cifar10_loaders(
        batch_size=BATCH_SIZE, augment=True,
    )

    model = CifarLSTM().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Bidirectional   : {BIDIRECTIONAL}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True,
    )

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
            torch.save(model.state_dict(), os.path.join(model_dir, "lstm_best.pth"))

    print(f"\n🏆 Best Test Accuracy: {best_acc:.2f}%")
    print("ℹ️  LSTM should outperform the vanilla RNN from Module 2")
    print("   thanks to its gated memory cell.\n")

    # ── Final evaluation ──
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "lstm_best.pth"), weights_only=True,
    ))
    _, final_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device,
    )
    print(f"📋 Final Test Accuracy (best checkpoint): {final_acc:.2f}%\n")

    # ── Visualisations ──
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        title="LSTM Training Curves", filename="lstm_training_curves.png",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="LSTM Confusion Matrix", filename="lstm_confusion_matrix.png",
    )
    plot_per_class_accuracy(
        y_true, y_pred, class_names,
        title="LSTM Per-Class Accuracy", filename="lstm_per_class_accuracy.png",
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
        title="LSTM Sample Predictions",
        filename="lstm_sample_predictions.png",
    )

    # ── Save ──
    final_path = os.path.join(model_dir, "lstm_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"💾 Final model saved → {final_path}")

    # ── ONNX export ──
    try:
        dummy = torch.randn(1, 3, 32, 32, device=device)
        onnx_path = os.path.join(model_dir, "lstm_model.onnx")
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"📦 ONNX model exported → {onnx_path}")
    except Exception as e:
        print(f"⚠️  ONNX export skipped: {e}")

    print("\n✅ LSTM training complete!  Check outputs/ for plots.")


if __name__ == "__main__":
    main()
