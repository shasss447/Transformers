import torch.nn as nn
from utils import MultiHeadAttention,PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model,num_head,d_ff,dp):
        super(EncoderLayer).__init__()
        self.slf_attn=MultiHeadAttention(d_model,num_head)
        self.fd_frwd=PositionWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.drpout=nn.Dropout(dp)

    def forward(self,x,mask):
        attn_out=self.slf_attn(x,x,x,mask)
        x=self.norm1(x+self.drpout(attn_out))
        ff_out=self.fd_frwd(x)
        x=self.norm2(x+self.drpout(ff_out))
        return x