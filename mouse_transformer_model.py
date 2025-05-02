import torch
import torch.nn as nn


class MouseTransformer(nn.Module):
    def __init__(self, input_dim=3, model_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super(MouseTransformer, self).__init__()
        self.model_dim = model_dim

        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(model_dim, input_dim)  # output: (dt, dx, dy)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_fc(x)  # project input to model_dim
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_fc(x[:, -1, :])  # predict next step


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Example usage
if __name__ == '__main__':
    model = MouseTransformer()
    dummy_input = torch.randn(8, 20, 3)  # batch of 8 sequences, each with 20 steps of (dt, dx, dy)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [8, 3]
