import torch.nn as nn


class TransformerBlock(nn.Module):
    """ Standard Transformer block """

    def __init__(self, input_size, num_heads, dropout):
        super().__init__()
        assert input_size % num_heads == 0, "Input size must be divisible by num_heads!"
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.key   = nn.Linear(input_size, input_size)
        self.query = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.attn = nn.MultiheadAttention(input_size, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.GELU(),
            nn.Linear(4 * input_size, input_size),
            nn.Dropout(dropout),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = x + self.attn(
                    self.query(self.ln1(x)),
                    self.key(self.ln1(x)),
                    self.value(self.ln1(x)),
                    need_weights=False,
                )[0]
        x = x + self.mlp(self.ln2(x))
        return x

