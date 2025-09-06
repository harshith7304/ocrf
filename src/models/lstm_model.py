import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T) long
        emb = self.embedding(x)  # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, H)
        if self.lstm.bidirectional:
            # concat last layer's forward & backward
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last).squeeze(1)  # (B,)
        return logits
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        # x: (B, T)
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)
        # Use last hidden from both directions
        h_cat = torch.cat([h[-2], h[-1]], dim=1) if self.lstm.bidirectional else h[-1]
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat).squeeze(1)
        return logits
