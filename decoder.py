import torch.nn as nn
from utils import MultiHeadAttention,PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model,num_head,d_ff,dp):
        super(DecoderLayer).__init__()
        self.slf_attn=MultiHeadAttention(d_model,num_head)
        self.crs_attn=MultiHeadAttention(d_model,num_head)
        self.fd_frwd=PositionWiseFeedForward(d_model,d_ff)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dp)

    def forward(self,x,enc_out,src_mask,tgt_mask):
        attn_out=self.slf_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout(attn_out))
        attn_out=self.crs_attn(x,enc_out,enc_out,src_mask)
        x=self.norm2(x+self.dropout(attn_out))
        ff_out=self.fd_frwd(x)
        x=self.norm3(x+self.dropout(ff_out))
        return x
