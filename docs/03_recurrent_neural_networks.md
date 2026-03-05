# 🔄 Chapter 3 — Recurrent Neural Networks (RNN)

<div align="center">

*"RNNs give neural networks the ability to remember — they process sequences one step at a time, carrying memory forward."*

</div>

---

## 📑 Table of Contents

1. [What is an RNN?](#-what-is-an-rnn)
2. [Why Do We Need RNNs?](#-why-do-we-need-rnns)
3. [RNN Architecture — Step by Step](#-rnn-architecture--step-by-step)
4. [The Hidden State — Memory of the Network](#-the-hidden-state--memory-of-the-network)
5. [Mathematical Foundation](#-mathematical-foundation)
6. [Unrolling an RNN Through Time](#-unrolling-an-rnn-through-time)
7. [Backpropagation Through Time (BPTT)](#-backpropagation-through-time-bptt)
8. [Types of RNN Architectures](#-types-of-rnn-architectures)
9. [The Vanishing Gradient Problem](#-the-vanishing-gradient-problem)
10. [RNN for Computer Vision](#-rnn-for-computer-vision)
11. [Our Implementation](#-our-implementation)

---

## 🧠 What is an RNN?

A **Recurrent Neural Network (RNN)** is a neural network designed to process **sequential data** — data where order matters. Unlike feedforward networks that process each input independently, RNNs maintain a **hidden state** that carries information from previous steps.

```mermaid
graph LR
    subgraph "Feedforward Network"
        A1["Input"] --> B1["Hidden"] --> C1["Output"]
    end

    subgraph "Recurrent Network"
        A2["Input_t"] --> B2["Hidden"]
        B2 -->|"Memory loop"| B2
        B2 --> C2["Output_t"]
    end

    style B1 fill:#2196F3,stroke:#333,color:#fff
    style B2 fill:#FF9800,stroke:#333,color:#fff
```

### Key Insight

> **RNNs process data step by step, remembering what they've seen before.** Each output depends on both the current input AND all previous inputs.

---

## ❓ Why Do We Need RNNs?

Regular neural networks (CNNs, MLPs) have a fundamental limitation: **they can't handle variable-length sequences or remember past context.**

```mermaid
graph TD
    Q{"What type of data?"}
    Q -->|Fixed size, Independent| A["Use CNN / MLP"]
    Q -->|Variable length, Sequential| B["Use RNN"]

    A --> A1["Single images\nTabular data"]
    B --> B1["Text, Speech\nTime series\nVideo frames"]

    style Q fill:#9C27B0,stroke:#333,color:#fff
    style A fill:#2196F3,stroke:#333,color:#fff
    style B fill:#FF9800,stroke:#333,color:#fff
```

### Sequential Data Examples

| Data Type | Why Order Matters |
|-----------|-------------------|
| **Text** | "Dog bites man" ≠ "Man bites dog" |
| **Speech** | Sound waveform is a time sequence |
| **Video** | Frames in order tell a story |
| **Stock Prices** | Yesterday's price affects today's analysis |
| **Image Rows** | Processing an image row-by-row as a sequence |

---

## 🏗️ RNN Architecture — Step by Step

### The Single RNN Cell

```
                    ┌───────────────────────┐
                    │                       │
     h_{t-1} ─────►│     RNN Cell          ├─────► h_t
                    │                       │
     x_t ─────────►│  h_t = tanh(W_hh·h_{t-1} + W_xh·x_t + b) │
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                             y_t (output)
```

```mermaid
graph LR
    A["h_{t-1}\nPrevious\nHidden State"] --> C["RNN Cell\ntanh(W·h + W·x + b)"]
    B["x_t\nCurrent\nInput"] --> C
    C --> D["h_t\nNew\nHidden State"]
    C --> E["y_t\nOutput"]

    style A fill:#FF9800,stroke:#333,color:#fff
    style B fill:#4CAF50,stroke:#333,color:#fff
    style C fill:#2196F3,stroke:#333,color:#fff
    style D fill:#FF9800,stroke:#333,color:#fff
    style E fill:#F44336,stroke:#333,color:#fff
```

**Three key components:**
1. **Input `x_t`** — Current data at time step `t`
2. **Hidden state `h_t`** — Network's memory, passed to next step
3. **Output `y_t`** — Prediction at the current step

---

## 💾 The Hidden State — Memory of the Network

The hidden state is the RNN's memory. It accumulates information from all previous inputs in the sequence.

```mermaid
sequenceDiagram
    participant x1 as x₁(Input)
    participant h1 as h₁(Hidden)
    participant x2 as x₂(Input)
    participant h2 as h₂(Hidden)
    participant x3 as x₃(Input)
    participant h3 as h₃(Hidden)

    Note over h1: Knows about x₁
    x1->>h1: Process
    h1->>h2: Pass memory
    Note over h2: Knows about x₁, x₂
    x2->>h2: Process
    h2->>h3: Pass memory
    Note over h3: Knows about x₁, x₂, x₃
    x3->>h3: Process
```

### Hidden State Analogy

Think of reading a book:

```
Page 1: "Harry received a letter."
  → Your brain remembers: [Harry, letter]

Page 2: "He went to a magic school."
  → Your brain remembers: [Harry, letter, magic school]

Page 3: "He learned to fly on a broom."
  → Your brain remembers: [Harry, magic school, flying, broom]

Your brain doesn't re-read the entire book at each page.
It carries forward a SUMMARY (= hidden state).
```

---

## 📐 Mathematical Foundation

### Core Equations

At each time step $t$:

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

Where:
| Symbol | Meaning | Shape |
|--------|---------|-------|
| $x_t$ | Input at time $t$ | `(input_size,)` |
| $h_t$ | Hidden state at time $t$ | `(hidden_size,)` |
| $W_{xh}$ | Input-to-hidden weights | `(hidden_size, input_size)` |
| $W_{hh}$ | Hidden-to-hidden weights | `(hidden_size, hidden_size)` |
| $W_{hy}$ | Hidden-to-output weights | `(output_size, hidden_size)` |
| $b_h, b_y$ | Bias terms | `(hidden_size,)`, `(output_size,)` |
| $\tanh$ | Activation function | Squashes values to [-1, 1] |

### Why tanh?

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```
         1 ─────────────────╱─────
                          ╱
         0 ─────────────╱─────────
                      ╱
        -1 ─────────╱─────────────

Output range: [-1, 1]
• Centers outputs around 0
• Helps with gradient flow
```

---

## ⏰ Unrolling an RNN Through Time

An RNN can be "unrolled" to visualize how information flows across time steps:

```mermaid
graph LR
    subgraph "Time Step 1"
        x1["x₁"] --> rnn1["RNN"]
        h0["h₀ (zeros)"] --> rnn1
        rnn1 --> y1["y₁"]
        rnn1 --> h1["h₁"]
    end

    subgraph "Time Step 2"
        x2["x₂"] --> rnn2["RNN"]
        h1 --> rnn2
        rnn2 --> y2["y₂"]
        rnn2 --> h2["h₂"]
    end

    subgraph "Time Step 3"
        x3["x₃"] --> rnn3["RNN"]
        h2 --> rnn3
        rnn3 --> y3["y₃"]
        rnn3 --> h3["h₃"]
    end

    subgraph "Time Step T"
        xT["x_T"] --> rnnT["RNN"]
        h3 -->|"..."| rnnT
        rnnT --> yT["y_T"]
        rnnT --> hT["h_T"]
    end

    style rnn1 fill:#2196F3,stroke:#333,color:#fff
    style rnn2 fill:#2196F3,stroke:#333,color:#fff
    style rnn3 fill:#2196F3,stroke:#333,color:#fff
    style rnnT fill:#2196F3,stroke:#333,color:#fff
```

**Key Point:** All RNN cells share the SAME weights ($W_{xh}$, $W_{hh}$, $W_{hy}$). This is called **weight sharing** — it allows the network to generalize across time.

---

## 🔙 Backpropagation Through Time (BPTT)

Training an RNN requires propagating gradients **backward through every time step:**

```mermaid
graph RL
    subgraph "Backward Pass (Gradients flow right to left)"
        LT["Loss_T"] -->|∂L/∂y_T| yT["y_T"]
        yT --> hT["h_T"]
        hT --> h3["h₃"]
        h3 --> h2["h₂"]
        h2 --> h1["h₁"]
    end

    style LT fill:#F44336,stroke:#333,color:#fff
    style hT fill:#FF9800,stroke:#333,color:#fff
    style h3 fill:#FF9800,stroke:#333,color:#fff
    style h2 fill:#FF9800,stroke:#333,color:#fff
    style h1 fill:#FF9800,stroke:#333,color:#fff
```

### The Problem with BPTT

At each backward step, gradients are **multiplied** by the weight matrix $W_{hh}$:

$$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh} \cdot \text{diag}(\tanh'(z))$$

After $T$ steps:

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_{hh} \cdot \text{diag}(\tanh'(z_t))$$

If $|W_{hh}| < 1$: gradients **vanish** (shrink to zero)  
If $|W_{hh}| > 1$: gradients **explode** (grow unbounded)

This is the **vanishing/exploding gradient problem** — solved by LSTM (next chapter).

---

## 🔀 Types of RNN Architectures

```mermaid
graph TD
    A["RNN Variants"]
    A --> B["One-to-One\n(Standard NN)"]
    A --> C["One-to-Many\n(Image Captioning)"]
    A --> D["Many-to-One\n(Sentiment Analysis)"]
    A --> E["Many-to-Many\n(Translation)"]
    A --> F["Many-to-Many\n(Video Labeling)"]

    style A fill:#9C27B0,stroke:#333,color:#fff
    style B fill:#4CAF50,stroke:#333,color:#fff
    style C fill:#2196F3,stroke:#333,color:#fff
    style D fill:#FF9800,stroke:#333,color:#fff
    style E fill:#F44336,stroke:#333,color:#fff
    style F fill:#795548,stroke:#333,color:#fff
```

### Visual Comparison

```
One-to-One:     One-to-Many:    Many-to-One:    Many-to-Many:
                                                (same length)
    ○              ○─○─○─○         ○
    │              │                │              ○─○─○─○
    ○              ○                ○─○─○─○        │ │ │ │
    │                               │              ○─○─○─○
    ○              Image→Caption    Text→Sentiment Seq→Seq

Many-to-Many (different length):
    ○─○─○─○        ○─○─○─○─○─○
    │                    │ │ │ │ │ │
Encoder              Decoder
    (English)            (French)
```

### In This Project: Many-to-One

We process an image row-by-row (sequence of rows) → single classification:

```mermaid
graph LR
    A["Row 1"] --> R1["RNN"]
    A2["Row 2"] --> R2["RNN"]
    A3["Row 3"] --> R3["RNN"]
    AN["Row 32"] --> RN["RNN"]

    R1 --> R2
    R2 --> R3
    R3 -->|"..."| RN
    RN --> O["Class: Cat 🐱"]

    style R1 fill:#FF9800,stroke:#333,color:#fff
    style R2 fill:#FF9800,stroke:#333,color:#fff
    style R3 fill:#FF9800,stroke:#333,color:#fff
    style RN fill:#FF9800,stroke:#333,color:#fff
    style O fill:#F44336,stroke:#333,color:#fff
```

---

## 📉 The Vanishing Gradient Problem

### The Core Issue

```mermaid
graph LR
    A["h₁"] -->|"×W"| B["h₂"]
    B -->|"×W"| C["h₃"]
    C -->|"×W"| D["..."]
    D -->|"×W"| E["h₁₀₀"]

    subgraph "Gradient magnitude"
        G1["1.0"] --> G2["0.5"]
        G2 --> G3["0.25"]
        G3 --> G4["0.001"]
        G4 --> G5["≈ 0"]
    end

    style A fill:#4CAF50,stroke:#333,color:#fff
    style E fill:#F44336,stroke:#333,color:#fff
    style G1 fill:#4CAF50,stroke:#333,color:#fff
    style G5 fill:#F44336,stroke:#333,color:#fff
```

### What This Means in Practice

```
Sequence: "The cat, which was sitting on the warm blanket 
           near the fireplace in the cozy living room, was ___"

RNN tries to predict: "sleeping"

But it has FORGOTTEN "The cat" by the time it reaches "was ___"
because gradients vanished — early time steps barely update.
```

### Solutions

| Solution | How It Works |
|----------|-------------|
| **Gradient Clipping** | Cap gradients to max value (prevents explosion only) |
| **LSTM** | Gate mechanisms to control gradient flow ✅ |
| **GRU** | Simplified LSTM with fewer gates |
| **Attention** | Direct connections to all time steps |

**→ This is exactly why we move to LSTM in the next chapter!**

---

## 🖼️ RNN for Computer Vision

### Treating an Image as a Sequence

A 32×32 image can be treated as a sequence of 32 rows, each row being a vector of 32×3 = 96 values:

```mermaid
graph TD
    A["Image 32×32×3"] --> B["Split into rows"]
    B --> C["Row 1: 1×96"]
    B --> D["Row 2: 1×96"]
    B --> E["..."]
    B --> F["Row 32: 1×96"]

    C --> G["RNN\n(processes row by row)"]
    D --> G
    E --> G
    F --> G

    G --> H["Final hidden state\nh₃₂"]
    H --> I["Classifier\n→ 10 classes"]

    style A fill:#4CAF50,stroke:#333,color:#fff
    style G fill:#FF9800,stroke:#333,color:#fff
    style I fill:#F44336,stroke:#333,color:#fff
```

### Why This Works (Partially)

- RNN sees the image **top-to-bottom**
- It can learn spatial patterns along the vertical axis
- But it **loses horizontal spatial information** compared to CNNs
- **Accuracy is lower** than CNN — this is expected and educational

---

## 💻 Our Implementation

Our RNN model in `src/02_rnn/rnn_sequence_model.py`:

- **Input:** CIFAR-10 images treated as sequences of 32 rows
- **Sequence length:** 32 (one row per time step)
- **Input features per step:** 96 (32 pixels × 3 channels)
- **Hidden size:** 256
- **Architecture:** 2-layer RNN → Fully Connected → 10 classes

### Run It

```bash
python src/02_rnn/rnn_sequence_model.py
```

### Expected Output

```
Epoch [1/20], Loss: 1.8234, Accuracy: 34.56%
Epoch [2/20], Loss: 1.5123, Accuracy: 45.23%
...
Epoch [20/20], Loss: 0.9876, Accuracy: 62.34%

Test Accuracy: ~60-65%
(Lower than CNN due to spatial information loss — this is expected!)
```

---

## 🔑 Key Takeaways

1. **RNNs process sequential data** — one element at a time, maintaining memory
2. **Hidden state** carries information from all previous time steps
3. **Weight sharing** across time steps enables sequence generalization
4. **BPTT** trains RNNs but causes **vanishing/exploding gradients**
5. **Images can be treated as sequences** (rows), but CNNs are better for spatial data
6. **LSTM** (next chapter) solves the vanishing gradient problem

---

<div align="center">

**← Previous:** [CNN](02_convolutional_neural_networks.md) | **Next →** [LSTM — Long Short-Term Memory](04_long_short_term_memory.md)

</div>
