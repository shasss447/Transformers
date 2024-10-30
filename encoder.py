import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model,num_head,d_ff,dp):
        super(EncoderLayer).__init__()