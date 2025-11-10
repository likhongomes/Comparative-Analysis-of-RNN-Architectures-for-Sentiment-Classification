import os, argparse, shutil
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Helpers ----------

def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def _normalize_clip(val):
    s = str(val).strip().lower()
    return 1 if s in {"yes", "true", "1"} else 0


def _build_tag(row):
    arch = str(row['Model']).lower()           # rnn / lstm / bilstm
    act = str(row['Activation']).lower()       # relu / tanh / sigmoid
    opt = str(row['Optimizer']).lower()        # adam / sgd / rmsprop
    L = int(row['Seq Length'])
    clip = _normalize_clip(row['Grad Clipping'])
    return f"{arch}_{act}_{opt}_L{L}_clip{clip}"


def load_metrics(metrics_csv: str) -> pd.DataFrame:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"metrics_csv not found: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    required = ['Model','Activation','Optimizer','Seq Length','Grad Clipping','Accuracy','F1','Epoch Time (s)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in metrics.csv: {missing}")
    # Type normalization
    df['Seq Length'] = df['Seq Length'].astype(int)
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    df['F1'] = pd.to_numeric(df['F1'], errors='coerce')
    df['Epoch Time (s)'] = pd.to_numeric(df['Epoch Time (s)'], errors='coerce')
    # Tag to locate corresponding loss plots saved by train.py
    df['run_tag'] = df.apply(_build_tag, axis=1)
    return df


# ---------- Plots required by the spec ----------

def plot_seq_length_curves(df: pd.DataFrame, out_dir='results/plots'):
    _ensure_outdir(out_dir)
    for metric in ['Accuracy','F1']:
        plt.figure()
        grp = df.groupby('Seq Length')[metric].mean().sort_index()
        grp.plot(marker='o')
        plt.xlabel('Sequence Length')
        plt.ylabel(metric)
        plt.title(f'{metric} vs. Sequence Length (averaged)')
        path = os.path.join(out_dir, f'{metric.lower()}_vs_seq_length.png')
        plt.savefig(path, bbox_inches='tight')
        print('Saved', path)


def plot_group_bars(df: pd.DataFrame, group_col: str, metric: str, out_dir='results/plots'):
    _ensure_outdir(out_dir)
    plt.figure()
    grp = df.groupby(group_col)[metric].mean().sort_values(ascending=False)
    grp.plot(kind='bar')
    plt.ylabel(f'Mean {metric}')
    plt.title(f'{metric} by {group_col} (averaged)')
    path = os.path.join(out_dir, f'{metric.lower()}_by_{group_col.replace(" ", "_").lower()}.png')
    plt.savefig(path, bbox_inches='tight')
    print('Saved', path)


def copy_best_worst_loss_plots(df: pd.DataFrame, out_dir='results/plots', source_dir='results/plots'):
    """
    Uses the run_tag convention from train.py to locate loss plots for the
    best and worst Accuracy runs, then copies them to canonical filenames
    for easy inclusion in the report.
    """
    _ensure_outdir(out_dir)
    df_sorted = df.sort_values('Accuracy', ascending=False)
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]

    def _try_copy(tag, dest):
        src = os.path.join(source_dir, f"loss_{tag}.png")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, dest))
            print('Saved', os.path.join(out_dir, dest))
        else:
            print(f"Warning: could not find loss plot for tag '{tag}' at {src}. Did you run train.py for that row?")

    _try_copy(best['run_tag'], 'best_loss.png')
    _try_copy(worst['run_tag'], 'worst_loss.png')


# ---------- Text summaries & tables ----------

def summarize_best_worst(df: pd.DataFrame):
    df_sorted = df.sort_values('Accuracy', ascending=False)
    best = df_sorted.head(1)
    worst = df_sorted.tail(1)
    print("Best configuration:", best.to_string(index=False))
    print("Worst configuration:", worst.to_string(index=False))
    return best, worst


def export_leaderboard(df: pd.DataFrame, out_csv='results/leaderboard.csv', top_k=20):
    _ensure_outdir(os.path.dirname(out_csv) or '.')
    cols = ['Model','Activation','Optimizer','Seq Length','Grad Clipping','Accuracy','F1','Epoch Time (s)']
    board = df.sort_values('Accuracy', ascending=False)[cols].head(top_k)
    board.to_csv(out_csv, index=False)
    print('Saved', out_csv)


# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_csv', type=str, default='results/metrics.csv')
    parser.add_argument('--out_dir', type=str, default='results/plots')
    parser.add_argument('--no_extras', action='store_true', help='Only make required sequence-length and loss plots')
    args = parser.parse_args()

    df = load_metrics(args.metrics_csv)

    # Required plots
    plot_seq_length_curves(df, args.out_dir)
    copy_best_worst_loss_plots(df, out_dir=args.out_dir, source_dir=args.out_dir)

    # Optional extras: helpful for your report
    if not args.no_extras:
        for group in ['Model','Optimizer','Activation','Grad Clipping']:
            for met in ['Accuracy','F1']:
                plot_group_bars(df, group, met, out_dir=args.out_dir)
        export_leaderboard(df, out_csv='results/leaderboard.csv', top_k=20)
        summarize_best_worst(df)
    else:
        summarize_best_worst(df)