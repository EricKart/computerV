"""
============================================================
  VISUALIZATION UTILITY
  ---------------------
  Shared plotting functions used across all four modules.

  Every function saves its output to the `outputs/` folder
  so you can review results even after the script finishes.
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Consistent style across all plots ──────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Default output folder (project_root/outputs)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "outputs",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. TRAINING CURVES  (loss & accuracy per epoch)
# ============================================================
def plot_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    title: str = "Training Curves",
    filename: str = "training_curves.png",
):
    """
    Draw a 1×2 figure: loss curve on the left, accuracy on the right.

    Parameters
    ----------
    train_losses, test_losses : per-epoch average loss
    train_accs,   test_accs   : per-epoch accuracy  (0-100 %)
    title    : super-title for the whole figure
    filename : saved under  outputs/<filename>
    """

    epochs = range(1, len(train_losses) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: Loss ----
    ax_loss.plot(epochs, train_losses, "o-", label="Train Loss")
    ax_loss.plot(epochs, test_losses,  "s-", label="Test Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.legend()

    # ---- Right: Accuracy ----
    ax_acc.plot(epochs, train_accs, "o-", label="Train Acc")
    ax_acc.plot(epochs, test_accs,  "s-", label="Test Acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.legend()

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Saved training curves → {save_path}")


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
):
    """
    Plot a heatmap confusion matrix and save to disk.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    class_names : list of class name strings
    """

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,            # show numbers inside each cell
        fmt="d",               # integer format
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Saved confusion matrix → {save_path}")


# ============================================================
# 3. SAMPLE PREDICTIONS GRID
# ============================================================
def plot_sample_predictions(
    images: np.ndarray,
    true_labels: list[int],
    pred_labels: list[int],
    class_names: list[str],
    n: int = 16,
    title: str = "Sample Predictions",
    filename: str = "sample_predictions.png",
):
    """
    Show the first `n` images in a grid with TRUE vs PRED labels.
    Correct predictions are green; wrong ones are red.

    Parameters
    ----------
    images      : numpy array of shape (N, H, W, 3) with values in [0, 1]
    true_labels : ground-truth integer labels
    pred_labels : predicted integer labels
    n           : how many images to show (must be a perfect square ideally)
    """

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i]
        # Clip to [0, 1] for display (normalisation can push values outside)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")

        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        correct   = true_labels[i] == pred_labels[i]

        ax.set_title(
            f"T: {true_name}\nP: {pred_name}",
            fontsize=8,
            color="green" if correct else "red",
            fontweight="bold",
        )

    # Hide any unused subplots
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Saved sample predictions → {save_path}")


# ============================================================
# 4. PER-CLASS ACCURACY BAR CHART
# ============================================================
def plot_per_class_accuracy(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str = "Per-Class Accuracy",
    filename: str = "per_class_accuracy.png",
):
    """
    Horizontal bar chart showing accuracy for each class.
    """

    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy = diagonal / row sum
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    colours = sns.color_palette("viridis", len(class_names))
    bars = ax.barh(class_names, per_class_acc, color=colours)

    # Annotate each bar with the percentage
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%", va="center", fontsize=9,
        )

    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 110)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"📊 Saved per-class accuracy → {save_path}")


# ============================================================
# 5. DENORMALIZE IMAGES FOR DISPLAY
# ============================================================
def denormalize_cifar10(images_tensor):
    """
    Reverse the Normalize(mean, std) transform so images
    look correct when plotted with matplotlib.

    Parameters
    ----------
    images_tensor : torch.Tensor of shape (N, 3, 32, 32)

    Returns
    -------
    numpy array of shape (N, 32, 32, 3) with values clipped to [0, 1]
    """
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std  = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    imgs = images_tensor.cpu().numpy()
    imgs = imgs * std + mean                  # reverse normalisation
    imgs = np.clip(imgs, 0, 1)                # clip to valid range
    imgs = imgs.transpose(0, 2, 3, 1)         # NCHW → NHWC for matplotlib
    return imgs


# ============================================================
# QUICK SMOKE TEST
# ============================================================
if __name__ == "__main__":
    print("Visualization utilities loaded successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nAvailable functions:")
    print("  • plot_training_curves()")
    print("  • plot_confusion_matrix()")
    print("  • plot_sample_predictions()")
    print("  • plot_per_class_accuracy()")
    print("  • denormalize_cifar10()")
