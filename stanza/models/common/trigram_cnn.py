import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

# Inspired by spaCy NER architecture
class TrigramCNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
            nn.GELU(),
        )


    def forward(self, input):
        x = F.pad(input, (0, 0, 1, 1), 'constant', 0)
        x = x.unfold(1, 3, 1).mT.flatten(2, 3)
        x = self.mlp(x)
        return input + x
