# Run Setup Lean Guide (Classroom)

This file is a practical, classroom-first checklist for running the project live with students.

## 1. Before Class (5-10 min)

1. Open terminal at project root:

```powershell
cd C:\Users\dukea\OneDrive\Documents\CV
```

2. Pull latest changes:

```powershell
git pull
```

3. Create/update environment:

```powershell
.\setup.ps1
```

4. Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

5. Quick dependency check:

```powershell
python -c "import torch, torchvision, matplotlib; print('OK')"
```

## 2. In Class Startup (first 10 min)

1. Explain the learning order:
- CNN first (best image baseline)
- RNN next (sequence idea)
- LSTM next (fixes RNN memory issues)
- Combined model last (CNN + LSTM design)

2. Ask students to open these docs in order:
- `docs/01_introduction_to_computer_vision.md`
- `docs/02_convolutional_neural_networks.md`
- `docs/03_recurrent_neural_networks.md`
- `docs/04_long_short_term_memory.md`
- `docs/05_cnn_rnn_lstm_combined.md`

## 3. Model Run Commands (live demo)

Run each command from project root.

```powershell
python -m src.01_cnn.cnn_image_classifier
python -m src.02_rnn.rnn_sequence_model
python -m src.03_lstm.lstm_model
python -m src.04_combined.cnn_rnn_lstm_combined
```

## 4. What To Check After Each Run

1. Accuracy trend in terminal per epoch.
2. Generated plots in `outputs/`:
- training curves
- confusion matrix
- per-class accuracy
- sample predictions
3. Saved model files in `models/` (`.pth` and `.onnx` where exported).

## 5. Class Talking Points

1. Why CNN usually beats RNN on static image classification.
2. Why LSTM improves over vanilla RNN for sequence memory.
3. Why combined CNN + LSTM is strong for video/sequential vision tasks.

## 6. End-of-Class Wrap-Up

1. Show deployment path:
- `docs/06_azure_deployment_guide.md`

2. Show portfolio sharing path:
- `docs/07_linkedin_publishing_guide.md`

3. Student homework:
- Change one hyperparameter (learning rate, epochs, hidden size).
- Re-run one module.
- Compare metrics and explain the difference.

## 7. Fast Troubleshooting

1. If `python` is not found:
- reinstall Python 3.10+ and re-open terminal.

2. If activation is blocked in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

3. If CUDA is unavailable:
- training will run on CPU (slower but valid).

4. If imports fail:

```powershell
pip install -r requirements.txt
```

## 8. 90-Minute Suggested Class Plan

1. 0-10 min: setup + repo overview.
2. 10-25 min: CV intro + CNN concepts.
3. 25-45 min: run CNN and inspect outputs.
4. 45-60 min: RNN + LSTM concept comparison.
5. 60-75 min: run RNN and LSTM, compare metrics.
6. 75-85 min: run combined model.
7. 85-90 min: Azure deployment + LinkedIn next steps.
