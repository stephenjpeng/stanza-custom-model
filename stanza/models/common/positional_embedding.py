import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """Module implementing Attention is All You Need pos embedding"""

    def __init__(self, input_size, max_block_size):
        super().__init__()
        pos = torch.arange(max_block_size).unsqueeze(-1)
        denom = torch.pow(10000, 2 * (torch.arange(0, input_size, 2) / input_size))

        pos_emb = torch.zeros(1, max_block_size, input_size)
        import pdb
        pdb.set_trace()
        pos_emb[0, :, 0::2] = torch.sin(pos / denom)
        pos_emb[0, :, 1::2] = torch.cos(pos / denom)

        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x, word_mask):
        out = self.pos_emb[:, :x.size()[1], :]
        return (~word_mask).unsqueeze(2) * out

