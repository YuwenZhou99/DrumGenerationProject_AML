from torch import nn

class DrumRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_dim)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden
