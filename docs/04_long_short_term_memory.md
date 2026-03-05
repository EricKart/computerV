# 🧠 Chapter 4 — Long Short-Term Memory (LSTM)

<div align="center">

*"LSTMs are RNNs with superpowers — they choose what to remember, what to forget, and what to output."*

</div>

---

## 📑 Table of Contents

1. [What is an LSTM?](#-what-is-an-lstm)
2. [The Problem LSTM Solves](#-the-problem-lstm-solves)
3. [LSTM Architecture — The Big Picture](#-lstm-architecture--the-big-picture)
4. [The Cell State — Information Highway](#-the-cell-state--information-highway)
5. [Gate 1: Forget Gate](#-gate-1-forget-gate)
6. [Gate 2: Input Gate](#-gate-2-input-gate)
7. [Gate 3: Output Gate](#-gate-3-output-gate)
8. [Complete LSTM Equations](#-complete-lstm-equations)
9. [Step-by-Step Walkthrough](#-step-by-step-walkthrough)
10. [LSTM vs RNN Comparison](#-lstm-vs-rnn-comparison)
11. [LSTM Variants](#-lstm-variants)
12. [Our Implementation](#-our-implementation)

---

## 🧠 What is an LSTM?

**Long Short-Term Memory (LSTM)** is a special type of RNN invented by Hochreiter & Schmidhuber in 1997. It is specifically designed to learn **long-term dependencies** by using a gating mechanism that controls the flow of information.

```mermaid
graph TD
    subgraph "Standard RNN Cell"
        A1["Simple: just tanh"]
        A2["h_t = tanh(W·[h_{t-1}, x_t])"]
    end

    subgraph "LSTM Cell"
        B1["Complex: 3 gates + cell state"]
        B2["Forget Gate"]
        B3["Input Gate"]
        B4["Output Gate"]
        B5["Cell State (highway)"]
    end

    style A1 fill:#F44336,stroke:#333,color:#fff
    style B1 fill:#4CAF50,stroke:#333,color:#fff
    style B2 fill:#FF9800,stroke:#333,color:#fff
    style B3 fill:#2196F3,stroke:#333,color:#fff
    style B4 fill:#9C27B0,stroke:#333,color:#fff
    style B5 fill:#00BCD4,stroke:#333,color:#fff
```

---

## 🔥 The Problem LSTM Solves

### Vanishing Gradient in Standard RNN

```mermaid
graph LR
    subgraph "RNN: Gradient Flow"
        A1["h₁"] -->|"×0.9"| A2["h₂"]
        A2 -->|"×0.9"| A3["h₃"]
        A3 -->|"×0.9"| A4["h₁₀"]
        A4 -->|"×0.9"| A5["h₁₀₀"]
    end

    subgraph "Gradient Magnitude"
        G1["1.0 "] --> G2["0.35"]
        G2 --> G3["0.04"]
        G3 --> G4["≈0.00"]
    end

    style A1 fill:#4CAF50,stroke:#333,color:#fff
    style A5 fill:#F44336,stroke:#333,color:#fff
```

### LSTM Solution: Controlled Gradient Highway

```mermaid
graph LR
    subgraph "LSTM: Gradient Flow"
        B1["h₁"] -->|"Gate ✓"| B2["h₂"]
        B2 -->|"Gate ✓"| B3["h₃"]
        B3 -->|"Gate ✓"| B4["h₁₀"]
        B4 -->|"Gate ✓"| B5["h₁₀₀"]
    end

    subgraph "Gradient Magnitude"
        H1["1.0 "] --> H2["0.95"]
        H2 --> H3["0.82"]
        H3 --> H4["0.71"]
    end

    style B1 fill:#4CAF50,stroke:#333,color:#fff
    style B5 fill:#4CAF50,stroke:#333,color:#fff
```

**Key Insight:** The LSTM's cell state acts as a **highway** where gradients can flow unchanged, preventing them from vanishing.

---

## 🏗️ LSTM Architecture — The Big Picture

```mermaid
graph TD
    subgraph "LSTM Cell"
        direction TB
        CT1["C_{t-1}\n(Previous Cell State)"] -->|"Highway"| FORGET["× Forget Gate\n(what to discard)"]
        FORGET --> ADD["+ "]
        INPUT["× Input Gate\n(what to add)"] --> ADD
        ADD --> CT["C_t\n(New Cell State)"]
        
        CT --> TANH["tanh"]
        TANH --> OUTPUT["× Output Gate\n(what to output)"]
        OUTPUT --> HT["h_t\n(New Hidden State)"]
        
        HT1["h_{t-1}"] --> GATES["Gate\nComputations"]
        XT["x_t"] --> GATES
        
        GATES --> FORGET
        GATES --> INPUT
        GATES --> OUTPUT
    end

    style FORGET fill:#F44336,stroke:#333,color:#fff
    style INPUT fill:#4CAF50,stroke:#333,color:#fff
    style OUTPUT fill:#FF9800,stroke:#333,color:#fff
    style CT fill:#2196F3,stroke:#333,color:#fff
    style HT fill:#9C27B0,stroke:#333,color:#fff
```

### The Two State Vectors

| State | Purpose | Analogy |
|-------|---------|---------|
| **Cell State ($C_t$)** | Long-term memory | Your notebook — stores important facts |
| **Hidden State ($h_t$)** | Short-term / working memory | Your current thought — what you focus on now |

---

## 🛤️ The Cell State — Information Highway

The cell state runs along the entire chain, with only minor linear interactions. Information can flow along it **unchanged** — this is the key innovation.

```
Cell State = Information Highway
═══════════════════════════════════════════════
                  │                │
            × (Forget)       + (Input)
                  │                │
═══════════════════════════════════════════════
                              │
                         tanh + × (Output)
                              │
                         Hidden State
```

```mermaid
graph LR
    C0["C₀"] -->|"× forget"| P1["  "]
    P1 -->|"+ input"| C1["C₁"]
    C1 -->|"× forget"| P2["  "]
    P2 -->|"+ input"| C2["C₂"]
    C2 -->|"× forget"| P3["  "]
    P3 -->|"+ input"| C3["C₃"]

    style C0 fill:#2196F3,stroke:#333,color:#fff
    style C1 fill:#2196F3,stroke:#333,color:#fff
    style C2 fill:#2196F3,stroke:#333,color:#fff
    style C3 fill:#2196F3,stroke:#333,color:#fff
```

---

## 🚪 Gate 1: Forget Gate

> **"What information from the past should I throw away?"**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

```mermaid
graph LR
    A["h_{t-1}\n(previous hidden)"] --> C["σ (sigmoid)"]
    B["x_t\n(current input)"] --> C
    C --> D["f_t\n(forget vector)"]
    D --> E["× C_{t-1}"]

    style C fill:#F44336,stroke:#333,color:#fff
    style D fill:#F44336,stroke:#333,color:#fff
```

### How It Works

The sigmoid function outputs values between **0 and 1** for each number in the cell state:

```
f_t = 0.0  → COMPLETELY FORGET this information
f_t = 0.5  → PARTIALLY keep this information
f_t = 1.0  → COMPLETELY KEEP this information

Example:
Cell State:  [0.8,  0.3,  0.9,  0.1]  (old memories)
Forget Gate: [1.0,  0.0,  0.7,  1.0]  (which to keep)
Result:      [0.8,  0.0,  0.63, 0.1]  (after forgetting)
              ↑     ↑      ↑     ↑
             kept  forgot  faded  kept
```

### Real-World Analogy

```
Reading a story about a person:
 - "Alice is a doctor." → Remember: Alice=doctor
 - "Alice quit her job." → Forget Gate activates!
   → Forget: Alice=doctor
   → Now Alice's profession slot is empty for new info
```

---

## 🚪 Gate 2: Input Gate

> **"What new information should I store?"**

Two parts:
1. **What to update:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
2. **Candidate values:** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

```mermaid
graph TD
    A["h_{t-1}"] --> S["σ (sigmoid)"]
    B["x_t"] --> S
    A --> T["tanh"]
    B --> T

    S --> IT["i_t\n(what to update)"]
    T --> CT["C̃_t\n(candidate values)"]

    IT --> MUL["×"]
    CT --> MUL
    MUL --> ADD["+ to Cell State"]

    style S fill:#4CAF50,stroke:#333,color:#fff
    style T fill:#2196F3,stroke:#333,color:#fff
    style MUL fill:#FF9800,stroke:#333,color:#fff
```

### How It Works

```
Step 1: Decide WHICH positions to update (sigmoid → 0 to 1)
  i_t = [0.1, 0.9, 0.3, 0.8]
         don't  DO  don't  DO
         update     update

Step 2: Create candidate values (tanh → -1 to 1)
  C̃_t = [-0.5, 0.7, 0.2, -0.9]

Step 3: Multiply to get actual update
  i_t × C̃_t = [-0.05, 0.63, 0.06, -0.72]

Step 4: Add to cell state
  C_t = f_t × C_{t-1} + i_t × C̃_t
```

### Updated Cell State

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Where $\odot$ means element-wise multiplication.

---

## 🚪 Gate 3: Output Gate

> **"What should I output based on the current cell state?"**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

```mermaid
graph TD
    A["h_{t-1}"] --> S["σ (sigmoid)"]
    B["x_t"] --> S
    
    S --> OT["o_t\n(output filter)"]
    
    CT["C_t\n(cell state)"] --> TH["tanh"]
    TH --> MUL["×"]
    OT --> MUL
    
    MUL --> HT["h_t\n(hidden state output)"]

    style S fill:#FF9800,stroke:#333,color:#fff
    style OT fill:#FF9800,stroke:#333,color:#fff
    style CT fill:#2196F3,stroke:#333,color:#fff
    style HT fill:#9C27B0,stroke:#333,color:#fff
```

### How It Works

```
Cell State (contains everything we know):
  C_t = [0.8, -0.3, 0.95, 0.1]

Output Gate (what's relevant RIGHT NOW):
  o_t = [1.0, 0.0, 0.5, 0.0]
         yes   no  maybe  no

Hidden State (what we actually output):
  h_t = o_t × tanh(C_t)
      = [1.0, 0.0, 0.5, 0.0] × [0.66, -0.29, 0.74, 0.10]
      = [0.66, 0.0, 0.37, 0.0]

The cell REMEMBERS everything, but only OUTPUTS what's relevant.
```

---

## 📐 Complete LSTM Equations

All four equations in one view:

$$\boxed{f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)}$$ ← Forget Gate

$$\boxed{i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)}$$ ← Input Gate

$$\boxed{\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)}$$ ← Candidate

$$\boxed{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$ ← Cell State Update

$$\boxed{o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)}$$ ← Output Gate

$$\boxed{h_t = o_t \odot \tanh(C_t)}$$ ← Hidden State

```mermaid
graph TD
    subgraph "Complete LSTM Data Flow"
        IN["[h_{t-1}, x_t]"] --> FG["Forget Gate (σ)"]
        IN --> IG["Input Gate (σ)"]
        IN --> CG["Candidate (tanh)"]
        IN --> OG["Output Gate (σ)"]
        
        FG -->|"f_t"| MUL1["×"]
        CT1["C_{t-1}"] --> MUL1
        
        IG -->|"i_t"| MUL2["×"]
        CG -->|"C̃_t"| MUL2
        
        MUL1 --> PLUS["+"]
        MUL2 --> PLUS
        
        PLUS --> CT["C_t"]
        CT --> TANH["tanh"]
        
        OG -->|"o_t"| MUL3["×"]
        TANH --> MUL3
        
        MUL3 --> HT["h_t"]
    end

    style FG fill:#F44336,stroke:#333,color:#fff
    style IG fill:#4CAF50,stroke:#333,color:#fff
    style CG fill:#2196F3,stroke:#333,color:#fff
    style OG fill:#FF9800,stroke:#333,color:#fff
    style CT fill:#00BCD4,stroke:#333,color:#fff
    style HT fill:#9C27B0,stroke:#333,color:#fff
```

---

## 🚶 Step-by-Step Walkthrough

Let's trace data through an LSTM processing image row features:

```
Time step t=1: Processing Row 1 of image
══════════════════════════════════════════

Inputs:
  x₁ = [0.5, 0.3, 0.8, ...]  (Row 1 pixel values, 96 values)
  h₀ = [0, 0, 0, ...]         (Initialized to zeros)
  C₀ = [0, 0, 0, ...]         (Initialized to zeros)

Step 1 — Forget Gate:
  f₁ = σ(W_f · [h₀, x₁] + b_f)
  f₁ = [0.8, 0.2, 0.9, ...]   (mostly keeping, it's the first step)

Step 2 — Input Gate:
  i₁ = σ(W_i · [h₀, x₁] + b_i)
  i₁ = [0.6, 0.9, 0.4, ...]   (deciding what to store from row 1)
  
  C̃₁ = tanh(W_C · [h₀, x₁] + b_C)
  C̃₁ = [0.3, -0.7, 0.5, ...]  (candidate new information)

Step 3 — Update Cell State:
  C₁ = f₁ ⊙ C₀ + i₁ ⊙ C̃₁
  C₁ = [0.8×0 + 0.6×0.3, 0.2×0 + 0.9×(-0.7), ...]
  C₁ = [0.18, -0.63, ...]

Step 4 — Output Gate:
  o₁ = σ(W_o · [h₀, x₁] + b_o)
  o₁ = [0.7, 0.5, 0.3, ...]
  
  h₁ = o₁ ⊙ tanh(C₁)
  h₁ = [0.7×tanh(0.18), 0.5×tanh(-0.63), ...]
  h₁ = [0.12, -0.27, ...]

→ h₁ and C₁ are passed to the next time step (Row 2)
```

---

## ⚔️ LSTM vs RNN Comparison

```mermaid
graph TD
    subgraph "RNN"
        R1["Simple architecture"]
        R2["Fast to compute"]
        R3["❌ Forgets long-term"]
        R4["❌ Vanishing gradients"]
        R5["~60% on CIFAR-10"]
    end

    subgraph "LSTM"
        L1["Complex (3 gates)"]
        L2["Slower to compute"]
        L3["✅ Remembers long-term"]
        L4["✅ Stable gradients"]
        L5["~68% on CIFAR-10"]
    end

    style R3 fill:#F44336,stroke:#333,color:#fff
    style R4 fill:#F44336,stroke:#333,color:#fff
    style L3 fill:#4CAF50,stroke:#333,color:#fff
    style L4 fill:#4CAF50,stroke:#333,color:#fff
```

| Feature | RNN | LSTM |
|---------|-----|------|
| **Parameters** | 1 weight matrix | 4 weight matrices (4× more) |
| **Memory** | Single hidden state | Cell state + hidden state |
| **Long sequences** | Fails (>10-20 steps) | Handles 100+ steps |
| **Training speed** | Fast per step | Slower per step |
| **Gradient flow** | Multiplicative (vanishes) | Additive (preserved) |
| **Use cases** | Short sequences | Long sequences, complex patterns |

---

## 🔀 LSTM Variants

```mermaid
graph TD
    A["LSTM Variants"]
    A --> B["Peephole LSTM\nGates see cell state"]
    A --> C["GRU\nSimplified: 2 gates"]
    A --> D["Bi-LSTM\nForward + Backward"]
    A --> E["Stacked LSTM\nMultiple layers"]

    style A fill:#9C27B0,stroke:#333,color:#fff
    style B fill:#2196F3,stroke:#333,color:#fff
    style C fill:#4CAF50,stroke:#333,color:#fff
    style D fill:#FF9800,stroke:#333,color:#fff
    style E fill:#F44336,stroke:#333,color:#fff
```

### GRU (Gated Recurrent Unit) — Simplified LSTM

```mermaid
graph LR
    subgraph "LSTM (3 gates)"
        L1["Forget Gate"]
        L2["Input Gate"]
        L3["Output Gate"]
    end

    subgraph "GRU (2 gates)"
        G1["Reset Gate\n(≈ Forget)"]
        G2["Update Gate\n(≈ Input + Forget)"]
    end

    style L1 fill:#F44336,stroke:#333,color:#fff
    style L2 fill:#4CAF50,stroke:#333,color:#fff
    style L3 fill:#FF9800,stroke:#333,color:#fff
    style G1 fill:#F44336,stroke:#333,color:#fff
    style G2 fill:#4CAF50,stroke:#333,color:#fff
```

### Bidirectional LSTM

Processes the sequence in **both directions** for richer context:

```mermaid
graph LR
    subgraph "Forward LSTM →"
        F1["h₁→"] --> F2["h₂→"] --> F3["h₃→"]
    end

    subgraph "Backward LSTM ←"
        B3["h₃←"] --> B2["h₂←"] --> B1["h₁←"]
    end

    F3 --> OUT["Concatenate\n[h₃→, h₁←]"]
    B1 --> OUT

    style OUT fill:#9C27B0,stroke:#333,color:#fff
```

---

## 💻 Our Implementation

Our LSTM model in `src/03_lstm/lstm_model.py`:

- **Input:** CIFAR-10 images as sequences of 32 rows
- **Architecture:** 2-layer LSTM → Fully Connected → 10 classes
- **Key difference from RNN:** Uses LSTM cells with gating mechanisms
- **Expected accuracy:** ~65-70% (better than vanilla RNN)

### Run It

```bash
python src/03_lstm/lstm_model.py
```

### Expected Improvement Over RNN

```
RNN:  ~60-65% accuracy  (forgets early rows)
LSTM: ~65-70% accuracy  (remembers across all 32 rows)

The improvement comes from LSTM's ability to
remember information from Row 1 when processing Row 32.
```

---

## 🔑 Key Takeaways

1. **LSTM** solves the vanishing gradient problem with **gate mechanisms**
2. **Cell state** acts as an information highway for long-term memory
3. **Forget gate** decides what old information to discard
4. **Input gate** decides what new information to store
5. **Output gate** decides what to expose as the current output
6. **LSTM outperforms vanilla RNN** on tasks requiring long-term memory
7. **GRU** is a simpler alternative with comparable performance

---

<div align="center">

**← Previous:** [RNN](03_recurrent_neural_networks.md) | **Next →** [CNN + RNN + LSTM Combined](05_cnn_rnn_lstm_combined.md)

</div>
