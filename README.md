## README.md


# Comparative Analysis of RNN Architectures for Sentiment Classification

This repository implements and evaluates RNN, LSTM, and Bidirectional LSTM models for binary sentiment classification on IMDb (50k reviews), following your project spec.

## Setup (May not be necessary if using provided environment)
- Python >= 3.10
- Install deps:
  ```bash
  python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt


* First run will download IMDb via `torchtext` into `.data/`.

## Reproducibility

We fix seeds in `train.py` via `torch.manual_seed(42)`, NumPy, and `random`.

## How to Run

Preprocess + single run (default RNN/RELU/Adam/seq=50, dropout=0.5, hidden=64):

```bash
python train.py --architecture rnn --activation relu --optimizer adam --seq_len 50 --epochs 5 --grad_clip_enable
```

Run the full sweep across required variations (can customize lists):

```bash
python train.py --sweep 
```

> Tip: Start with fewer epochs on CPU.

Generate plots and a text summary of best/worst:

```bash
python evaluate.py
```

## Expected Outputs

* `results/metrics.csv` with rows:
  `Model,Activation,Optimizer,Seq Length,Grad Clipping,Accuracy,F1,Epoch Time (s),Epochs,Hardware`
* Plots in `results/plots/`:

  * `accuracy_vs_seq_length.png`
  * `f1_vs_seq_length.png`
  * `loss_<MODEL>_<ACT>_<OPT>_L<SEQ>_clip<0|1>.png`

## Hardware Note

The code auto-detects CUDA; otherwise runs on CPU. For my experiments, I used an M1 Max MacBook Pro with 32GB RAM

## Project Report

The full report is in `report.pdf`, covering methodology, results, and analysis.

````
