
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np

# def train_model(embedding_layer_size=100,architecture="RNN",activation="sigmoid", gradient_clipping=True,optimizer_choice="Adam"):

#     if optimizer_choice == "Adam":
#         optimizer = keras.optimizers.Adam(
#             learning_rate=0.01,
#             clipnorm=1.0 if gradient_clipping else None
#         )
#     if optimizer_choice == "SGD":
#         optimizer = keras.optimizers.SGD(
#             learning_rate=0.01,
#             clipnorm=1.0 if gradient_clipping else None
#         )
#     if optimizer_choice == "RMSProp":
#         optimizer = keras.optimizers.SGD(
#             learning_rate=0.01,
#             clipnorm=1.0 if gradient_clipping else None
#         )


#     # ----- Toy data (binary tokens), same idea as PyTorch example
#     def make_data(n=3000, T=20, threshold=10):
#         X = np.random.randint(0, 2, size=(n, T))
#         y = (X.sum(axis=1) >= threshold).astype("int32")
#         return X, y

#     X_train, y_train = make_data(4000, 25, threshold=12)
#     X_val,   y_val   = make_data(800,  25, threshold=12)

#     vocab_size = 2  # tokens {0,1}

#     inputs = keras.Input(shape=(X_train.shape[1],), dtype="int32")
#     x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_layer_size)(inputs)

#     if architecture == "RNN":
#         x = layers.SimpleRNN(64, return_sequences=True)(x)
#         x = layers.SimpleRNN(64)(x)

#     if architecture == "LSTM":
#         x = layers.LSTM(64, return_sequences=True)(x)  # 1st LSTM layer
#         x = layers.LSTM(64)(x) # 2nd layer

#     if architecture == "BLSTM":
#         x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
#         x = layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)

#     outputs = layers.Dense(1, activation=activation)(x)

#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
#     model.summary()

#     model.fit(X_train, y_train, batch_size=32, epochs=8,
#             validation_data=(X_val, y_val))
    

# train_model()










# import os, time, argparse, json, itertools
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import Adam, SGD, RMSprop
# from sklearn.metrics import accuracy_score, f1_score
# from utils import set_seed, get_device, epoch_time, compute_metrics, save_row
# from preprocess import preprocess_and_cache, make_dataloaders
# from models import RNNClassifier


# def get_optimizer(name, params, lr):
#     name = name.lower()
#     if name == 'adam':
#         return Adam(params, lr=lr)
#     if name == 'sgd':
#         return SGD(params, lr=lr, momentum=0.0)
#     if name == 'rmsprop':
#         return RMSprop(params, lr=lr)
#     raise ValueError('Unknown optimizer: ' + name)


# def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
#     model.train()
#     losses = []
#     all_y, all_p = [], []
#     start = time.time()
#     for X, y in loader:
#         X = X.to(device)
#         y = y.to(device)
#         optimizer.zero_grad()
#         p = model(X)
#         loss = criterion(p, y)
#         loss.backward()
#         if grad_clip is not None and grad_clip > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#         optimizer.step()
#         losses.append(loss.item())
#         all_y.append(y.detach().cpu().numpy())
#         all_p.append((p.detach().cpu().numpy() >= 0.5).astype(np.int32))
#     end = time.time()
#     y_true = np.concatenate(all_y)
#     y_pred = np.concatenate(all_p)
#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')
#     return np.mean(losses), acc, f1, epoch_time(start, end)


# def evaluate(model, loader, criterion, device):
#     model.eval()
#     losses = []
#     all_y, all_p = [], []
#     with torch.no_grad():
#         for X, y in loader:
#             X = X.to(device)
#             y = y.to(device)
#             p = model(X)
#             loss = criterion(p, y)
#             losses.append(loss.item())
#             all_y.append(y.cpu().numpy())
#             all_p.append((p.cpu().numpy() >= 0.5).astype(np.int32))
#     y_true = np.concatenate(all_y)
#     y_pred = np.concatenate(all_p)
#     acc = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, average='macro')
#     return np.mean(losses), acc, f1


# def run_experiment(args):
#     set_seed(42)
#     device = get_device()

#     # --- Ensure CSV exists
#     csv_path = os.path.join(args.data_dir, "IMDB Dataset.csv")
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"Expected dataset file at {csv_path}")

#     # --- Preprocess/cache & build loaders
#     train_cache = os.path.join(args.data_dir, "train.pt")
#     test_cache = os.path.join(args.data_dir, "test.pt")

#     if not (os.path.exists(train_cache) and os.path.exists(test_cache)):
#         print("Preprocessing IMDb CSV dataset ...")
#         train_loader, test_loader = preprocess_and_cache(
#             seq_len=args.seq_len, vocab_size=args.vocab_size, out_dir=args.data_dir
#         )
#     else:
#         print("Loading preprocessed data ...")
#         train_loader, test_loader = make_dataloaders(
#             seq_len=args.seq_len, data_dir=args.data_dir, batch_size=args.batch_size
#         )

#     # --- Load vocab size
#     vocab_path = os.path.join(args.data_dir, "vocab.json")
#     with open(vocab_path) as f:
#         stoi = json.load(f)
#     vocab_size = max(stoi.values()) + 1

#     # --- Build model
#     arch = args.architecture
#     bidir = (arch == 'bilstm')
#     model = RNNClassifier(
#         vocab_size=vocab_size,
#         emb_dim=args.emb_dim,
#         hidden_size=args.hidden_size,
#         num_layers=2,
#         dropout=args.dropout,
#         architecture='lstm' if arch in ['lstm', 'bilstm'] else 'rnn',
#         activation=args.activation,
#         bidirectional=bidir
#     ).to(device)

#     # --- Optimizer & loss
#     optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr)
#     criterion = nn.BCELoss()

#     # --- Train loop
#     os.makedirs('results/plots', exist_ok=True)
#     history = { 'train_loss': [], 'train_acc': [], 'train_f1': [], 'epoch_time': [] }
#     last_te_acc = last_te_f1 = None

#     for epoch in range(1, args.epochs + 1):
#         tr_loss, tr_acc, tr_f1, ep_time = train_one_epoch(
#             model, train_loader, optimizer, criterion, device,
#             grad_clip=args.grad_clip if args.grad_clip_enable else None
#         )
#         te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion, device)

#         history['train_loss'].append(tr_loss)
#         history['train_acc'].append(tr_acc)
#         history['train_f1'].append(tr_f1)
#         history['epoch_time'].append(ep_time)

#         last_te_acc, last_te_f1 = te_acc, te_f1
#         print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | test_acc={te_acc:.4f} f1={te_f1:.4f} | time={ep_time:.1f}s")

#     # --- Save metrics row
#     row = {
#         'Model': args.architecture.upper(),
#         'Activation': args.activation,
#         'Optimizer': args.optimizer,
#         'Seq Length': args.seq_len,
#         'Grad Clipping': 'Yes' if args.grad_clip_enable else 'No',
#         'Accuracy': round(last_te_acc, 4) if last_te_acc is not None else None,
#         'F1': round(last_te_f1, 4) if last_te_f1 is not None else None,
#         'Epoch Time (s)': round(float(np.mean(history['epoch_time'])), 2) if history['epoch_time'] else None,
#         'Epochs': args.epochs,
#         'Hardware': 'CUDA' if torch.cuda.is_available() else 'CPU'
#     }
#     save_row('results/metrics.csv', row, header_order=[
#         'Model','Activation','Optimizer','Seq Length','Grad Clipping','Accuracy','F1','Epoch Time (s)','Epochs','Hardware'
#     ])

#     # --- Save training loss curve image (used by evaluate.py)
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.plot(range(1, args.epochs + 1), history['train_loss'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Training Loss')
#     tag = f"{args.architecture.lower()}_{args.activation.lower()}_{args.optimizer.lower()}_L{args.seq_len}_clip{int(bool(args.grad_clip_enable))}"
#     plt.title(f'Training Loss – {tag}')
#     path = f"results/plots/loss_{tag}.png"
#     plt.savefig(path, bbox_inches='tight')
#     print('Saved', path)



# def sweep(args):
#     models = ['rnn','lstm','bilstm'] if args.models is None else args.models.split(',')
#     activations = ['relu','tanh','sigmoid'] if args.activations is None else args.activations.split(',')
#     optimizers = ['adam','sgd','rmsprop'] if args.optimizers is None else args.optimizers.split(',')
#     seq_lengths = [25,50,100] if args.seq_lengths is None else list(map(int, args.seq_lengths.split(',')))
#     clipping = [False, True] if args.clipping is None else [(c.strip().lower()=='true') for c in args.clipping.split(',')]

#     for m, a, o, L, c in itertools.product(models, activations, optimizers, seq_lengths, clipping):
#         print("\n=== RUN:", m, a, o, L, c, '===')
#         run_args = argparse.Namespace(**vars(args))
#         run_args.architecture = m
#         run_args.activation = a
#         run_args.optimizer = o
#         run_args.seq_len = L
#         run_args.grad_clip_enable = c
#         run_experiment(run_args)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # Fixed defaults matching spec
#     parser.add_argument('--emb_dim', type=int, default=100)
#     parser.add_argument('--hidden_size', type=int, default=64)
#     parser.add_argument('--dropout', type=float, default=0.5)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=5)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--vocab_size', type=int, default=10000)

#     # Variations
#     parser.add_argument('--architecture', type=str, default='rnn', choices=['rnn','lstm','bilstm'])
#     parser.add_argument('--activation', type=str, default='relu', choices=['relu','tanh','sigmoid'])
#     parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd','rmsprop'])
#     parser.add_argument('--seq_len', type=int, default=50, choices=[25,50,100])
#     parser.add_argument('--grad_clip_enable', action='store_true')
#     parser.add_argument('--grad_clip', type=float, default=1.0)

#     # Paths
#     parser.add_argument('--data_dir', type=str, default='data')

#     # Sweep helper
#     parser.add_argument('--sweep', action='store_true')
#     parser.add_argument('--models', type=str, default=None)
#     parser.add_argument('--activations', type=str, default=None)
#     parser.add_argument('--optimizers', type=str, default=None)
#     parser.add_argument('--seq_lengths', type=str, default=None)
#     parser.add_argument('--clipping', type=str, default=None)

#     args = parser.parse_args()

#     if args.sweep:
#         sweep(args)
#     else:
#         run_experiment(args)





import os, time, argparse, json, itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from sklearn.metrics import accuracy_score, f1_score
from utils import set_seed, get_device, epoch_time, compute_metrics, save_row
from preprocess import preprocess_and_cache, make_dataloaders
from models import RNNClassifier


def get_optimizer(name, params, lr):
    name = name.lower()
    if name == 'adam':
        return Adam(params, lr=lr)
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.0)
    if name == 'rmsprop':
        return RMSprop(params, lr=lr)
    raise ValueError('Unknown optimizer: ' + name)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    losses = []
    all_y, all_p = [], []
    start = time.time()
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        p = model(X)
        loss = criterion(p, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        all_y.append(y.detach().cpu().numpy())
        all_p.append((p.detach().cpu().numpy() >= 0.5).astype(np.int32))
    end = time.time()
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return np.mean(losses), acc, f1, epoch_time(start, end)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y, all_p = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            p = model(X)
            loss = criterion(p, y)
            losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_p.append((p.cpu().numpy() >= 0.5).astype(np.int32))
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return np.mean(losses), acc, f1


def run_experiment(args):
    set_seed(42)
    device = get_device()

    # --- Ensure CSV exists
    csv_path = os.path.join(args.data_dir, "IMDB Dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected dataset file at {csv_path}")

    # --- Preprocess/cache & build loaders
    train_cache = os.path.join(args.data_dir, "train.pt")
    test_cache = os.path.join(args.data_dir, "test.pt")

    if not (os.path.exists(train_cache) and os.path.exists(test_cache)):
        print("Preprocessing IMDb CSV dataset ...")
        train_loader, test_loader = preprocess_and_cache(
            seq_len=args.seq_len, vocab_size=args.vocab_size, out_dir=args.data_dir
        )
    else:
        print("Loading preprocessed data ...")
        train_loader, test_loader = make_dataloaders(
            seq_len=args.seq_len, data_dir=args.data_dir, batch_size=args.batch_size
        )

    # --- Load vocab size
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    with open(vocab_path) as f:
        stoi = json.load(f)
    vocab_size = max(stoi.values()) + 1

    # --- Build model
    arch = args.architecture
    bidir = (arch == 'bilstm')
    model = RNNClassifier(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=args.dropout,
        architecture='lstm' if arch in ['lstm', 'bilstm'] else 'rnn',
        activation=args.activation,
        bidirectional=bidir
    ).to(device)

    # --- Optimizer & loss
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # --- Train loop
    os.makedirs('results/plots', exist_ok=True)
    history = { 'train_loss': [], 'train_acc': [], 'train_f1': [], 'epoch_time': [] }
    last_te_acc = last_te_f1 = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1, ep_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=args.grad_clip if args.grad_clip_enable else None
        )
        te_loss, te_acc, te_f1 = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_f1'].append(tr_f1)
        history['epoch_time'].append(ep_time)

        last_te_acc, last_te_f1 = te_acc, te_f1
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | test_acc={te_acc:.4f} f1={te_f1:.4f} | time={ep_time:.1f}s")

    # --- Save metrics row
    row = {
        'Model': args.architecture.upper(),
        'Activation': args.activation,
        'Optimizer': args.optimizer,
        'Seq Length': args.seq_len,
        'Grad Clipping': 'Yes' if args.grad_clip_enable else 'No',
        'Accuracy': round(last_te_acc, 4) if last_te_acc is not None else None,
        'F1': round(last_te_f1, 4) if last_te_f1 is not None else None,
        'Epoch Time (s)': round(float(np.mean(history['epoch_time'])), 2) if history['epoch_time'] else None,
        'Epochs': args.epochs,
        'Hardware': 'CUDA' if torch.cuda.is_available() else 'CPU'
    }
    save_row('results/metrics.csv', row, header_order=[
        'Model','Activation','Optimizer','Seq Length','Grad Clipping','Accuracy','F1','Epoch Time (s)','Epochs','Hardware'
    ])

    # --- Save training loss curve image (used by evaluate.py)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, args.epochs + 1), history['train_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    tag = f"{args.architecture.lower()}_{args.activation.lower()}_{args.optimizer.lower()}_L{args.seq_len}_clip{int(bool(args.grad_clip_enable))}"
    plt.title(f'Training Loss – {tag}')
    path = f"results/plots/loss_{tag}.png"
    plt.savefig(path, bbox_inches='tight')
    print('Saved', path)


def sweep(args):
    models = ['rnn','lstm','bilstm'] if args.models is None else args.models.split(',')
    activations = ['relu','tanh','sigmoid'] if args.activations is None else args.activations.split(',')
    optimizers = ['adam','sgd','rmsprop'] if args.optimizers is None else args.optimizers.split(',')
    seq_lengths = [25,50,100] if args.seq_lengths is None else list(map(int, args.seq_lengths.split(',')))
    clipping = [False, True] if args.clipping is None else [(c.strip().lower()=='true') for c in args.clipping.split(',')]

    for m, a, o, L, c in itertools.product(models, activations, optimizers, seq_lengths, clipping):
        print("=== RUN:", m, a, o, L, c, '===')
        run_args = argparse.Namespace(**vars(args))
        run_args.architecture = m
        run_args.activation = a
        run_args.optimizer = o
        run_args.seq_len = L
        run_args.grad_clip_enable = c
        run_experiment(run_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Fixed defaults matching spec
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--vocab_size', type=int, default=10000)

    # Variations
    parser.add_argument('--architecture', type=str, default='rnn', choices=['rnn','lstm','bilstm'])
    parser.add_argument('--activation', type=str, default='relu', choices=['relu','tanh','sigmoid'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd','rmsprop'])
    parser.add_argument('--seq_len', type=int, default=50, choices=[25,50,100])
    parser.add_argument('--grad_clip_enable', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Paths
    parser.add_argument('--data_dir', type=str, default='data')

    # Sweep helper
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--activations', type=str, default=None)
    parser.add_argument('--optimizers', type=str, default=None)
    parser.add_argument('--seq_lengths', type=str, default=None)
    parser.add_argument('--clipping', type=str, default=None)

    args = parser.parse_args()

    if args.sweep:
        sweep(args)
    else:
        run_experiment(args)