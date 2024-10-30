import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model,num_head,d_ff,dp):
        super(DecoderLayer).__init__()