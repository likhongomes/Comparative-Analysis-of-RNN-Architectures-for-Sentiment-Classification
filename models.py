
import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.5,
                 architecture: str = 'rnn', activation: str = 'tanh', bidirectional: bool = False):
        super().__init__()
        self.architecture = architecture
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        rnn_kwargs = dict(input_size=emb_dim,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True,
                          bidirectional=bidirectional)

        if architecture == 'rnn':
            nonlinearity = 'relu' if activation.lower() == 'relu' else 'tanh'
            self.rnn = nn.RNN(nonlinearity=nonlinearity, **rnn_kwargs)
        elif architecture == 'lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif architecture == 'bilstm' or (architecture == 'lstm' and bidirectional):
            self.rnn = nn.LSTM(**{**rnn_kwargs, 'bidirectional': True})
            self.bidirectional = True
        else:
            raise ValueError("architecture must be one of {'rnn','lstm','bilstm'}")

        rnn_out_dim = hidden_size * (2 if self.bidirectional else 1)

        # Optional post-RNN activation layer to satisfy activation sweep for LSTM/biLSTM
        act = activation.lower()
        if act == 'relu':
            self.post_act = nn.ReLU()
        elif act == 'tanh':
            self.post_act = nn.Tanh()
        elif act == 'sigmoid':
            self.post_act = nn.Sigmoid()
        else:
            raise ValueError("activation must be one of {'relu','tanh','sigmoid'}")

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(rnn_out_dim, 1)
        self.out_act = nn.Sigmoid()  # binary classification

    def forward(self, x):
        # x: (B, T)
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        # Use last time-step representation (assuming padded/truncated to same length)
        last = out[:, -1, :]
        last = self.post_act(last)
        last = self.dropout(last)
        logits = self.fc(last)
        prob = self.out_act(logits)
        return prob.squeeze(1)
