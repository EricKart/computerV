# Run Setup Comprehensive Guide (Classroom + Self-Learning)

This guide is the operational playbook for instructors and students to setup, run, validate, teach, and troubleshoot the project end-to-end.

---

## 1. Audience and Goals

Use this file if you want to:
- run all four modules reliably on classroom machines
- teach from baseline CNN to advanced combined architecture
- compare model behavior with measurable outputs
- finish with deployment and portfolio next steps

Expected outcomes:
- students can run every module independently
- students can explain what each model is doing
- students can interpret generated metrics and plots

---

## 2. Pre-Class Instructor Checklist (10-15 min)

1. Open terminal in project root:

```powershell
cd C:\Users\dukea\OneDrive\Documents\CV
```

2. Sync latest code:

```powershell
git pull
```

3. Create or refresh environment:

```powershell
.\setup.ps1
```

4. Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Verify dependencies:

```powershell
python --version
python -c "import torch, torchvision, matplotlib, seaborn, onnx; print('Environment OK')"
```

6. Confirm files/folders exist:
- `src/01_cnn/cnn_image_classifier.py`
- `src/02_rnn/rnn_sequence_model.py`
- `src/03_lstm/lstm_model.py`
- `src/04_combined/cnn_rnn_lstm_combined.py`
- `docs/01_*` to `docs/08_*`
- `models/` and `outputs/`

---

## 3. Quick Student Onboarding Script (first 8-10 min)

Say this sequence:
1. "Today we compare CNN, RNN, LSTM, and a combined model on the same dataset."
2. "CNN is our image baseline. RNN/LSTM teach sequence modeling. Combined model mimics video-style pipelines."
3. "After each run, we inspect evidence in `outputs/` and compare behavior."

Ask students to open:
- `docs/01_introduction_to_computer_vision.md`
- `docs/02_convolutional_neural_networks.md`
- `docs/03_recurrent_neural_networks.md`
- `docs/04_long_short_term_memory.md`
- `docs/05_cnn_rnn_lstm_combined.md`

---

## 4. Environment Setup for Students (detailed)

### Windows PowerShell

```powershell
git clone https://github.com/EricKart/computerV.git
cd computerV
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
git clone https://github.com/EricKart/computerV.git
cd computerV
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Why these commands matter
- `setup.ps1` / `setup.sh` creates isolated environment (`venv`) and installs all packages
- activation makes sure commands use project-local dependencies
- this prevents "works on my machine" package conflicts

---

## 5. Run Order and Commands

Run each from project root with active `venv`:

```powershell
python -m src.01_cnn.cnn_image_classifier
python -m src.02_rnn.rnn_sequence_model
python -m src.03_lstm.lstm_model
python -m src.04_combined.cnn_rnn_lstm_combined
```

Recommended order:
1. CNN
2. RNN
3. LSTM
4. Combined CNN + LSTM

Reason:
- starts from strongest static-image baseline
- shows why vanilla RNN struggles
- demonstrates LSTM memory improvement
- ends with realistic hybrid architecture

---

## 6. What Each Module Is Doing

| Module | Input representation | Core model idea | Typical behavior |
|--------|----------------------|-----------------|------------------|
| CNN | 32x32 image tensor | Convolution learns spatial patterns | Strong image baseline |
| RNN | image rows as sequence | Recurrence across row steps | weaker on static image tasks |
| LSTM | image rows as sequence | gated memory for longer dependencies | better than vanilla RNN |
| Combined | image -> patches -> sequence | CNN features + LSTM sequence modeling | strong for sequence-like visual tasks |

---

## 7. Output Validation Checklist After Every Run

Terminal checks:
- epoch logs are progressing (loss generally down, accuracy generally up)
- run completes without runtime errors

File checks:
1. `outputs/*training_curves*.png`
2. `outputs/*confusion_matrix*.png`
3. `outputs/*per_class_accuracy*.png`
4. `outputs/*sample_predictions*.png`
5. `models/*best.pth` and `models/*final.pth`
6. optional ONNX file: `models/*.onnx`

Discussion checks:
- Can students explain confusion matrix meaning?
- Can they identify hardest classes from per-class accuracy chart?

---

## 8. Class Delivery Plans

### 60-minute plan

1. 0-8 min: setup verification
2. 8-20 min: intro + CNN concepts
3. 20-35 min: run CNN and read outputs
4. 35-45 min: RNN vs LSTM theory
5. 45-55 min: run LSTM and compare
6. 55-60 min: summary and assignment

### 90-minute plan

1. 0-10 min: setup + repo map
2. 10-25 min: intro + CNN theory
3. 25-45 min: run CNN + evaluate outputs
4. 45-60 min: RNN and LSTM concept deep dive
5. 60-75 min: run RNN and LSTM + compare
6. 75-85 min: run combined model
7. 85-90 min: Azure + LinkedIn next steps

---

## 9. Troubleshooting Matrix

| Problem | Symptom | Fix |
|---------|---------|-----|
| Python missing | `python is not recognized` | install Python 3.10+ and reopen terminal |
| Activation blocked | script execution error in PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Missing imports | `ModuleNotFoundError` | activate `venv`; run `pip install -r requirements.txt` |
| Slow training | very long epoch times | reduce `EPOCHS`, reduce `BATCH_SIZE`, use GPU if available |
| OOM on GPU | CUDA out-of-memory | lower `BATCH_SIZE` (e.g., 64 -> 32 -> 16) |
| No GPU detected | running on CPU | expected fallback; continue or install CUDA drivers |

---

## 10. Teaching Prompts (for student understanding)

Ask these during class:
1. Why does CNN usually outperform RNN on static image classification?
2. What does LSTM remember that RNN tends to forget?
3. Which classes are most confused in your confusion matrix, and why?
4. When would you choose combined CNN + LSTM in a real product?

---

## 11. End-of-Class Next Steps

1. Deployment extension:
- `docs/06_azure_deployment_guide.md`

2. Career/portfolio extension:
- `docs/07_linkedin_publishing_guide.md`

3. Homework options:
- tune one hyperparameter and report delta
- compare any two modules with evidence from outputs
- export ONNX and document inference test steps

---

## 12. Teaching Assistant Checklist

Before class:
- confirm instructor machine runs all four modules
- verify internet access for dataset download
- pre-open docs and terminal tabs

During class:
- monitor failed setup machines first
- check environment activation errors quickly
- collect questions around confusion matrix interpretation

After class:
- archive best student outputs
- collect top three recurring issues for next session improvement
