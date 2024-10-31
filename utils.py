import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_head):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_head==0

        self.d_model=d_model
        self.num_head=num_head
        self.d_k=d_model

        self.wq=nn.Linear(d_model,d_model)
        self.wk=nn.Linear(d_model,d_model)
        self.wv=nn.Linear(d_model,d_model)
        self.wo=nn.Linear(d_model,d_model)

    def s_dp_attn(self,q,k,v,mask=None):
        attn_sc=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:
            attn_sc=attn_sc.masked_fill(mask==0,-1e9)
        
        attn_pb=torch.softmax(attn_sc,dim=-1)
        output=torch.matmul(attn_pb,v)
        return output

    def splt_hd(self,x):
        bt_sz,seq_len,d_m=x.size()
        return x.view(bt_sz,seq_len,self.num_head,self.d_k).transpose(1,2)

    def cmb_hd(self,x):
        bt_sz,_,seq_len,d_k=x.size()
        return x.transpose(1,2).contiguous().view(bt_sz,seq_len,self.d_model)

    def forward(self,q,k,v,mask=None):
        q=self.splt_hd(self.wq(q))
        k=self.splt_hd(self.wk(k))
        v=self.splt_hd(self.wv(v))

        attn_out=self.s_dp_attn(q,k,v,mask)
        output=self.wo(self.comb_hd(attn_out))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.relu=nn.ReLU()

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,mx_sq_len):
        super(PositionalEncoding,self).__init__()

        pe=torch.zeros(mx_sq_len,d_model)
        pos=torch.arange(0,mx_sq_len,dtype=torch.float).unsqueeze(1)
        dv_trm=torch.exp(torch.arange(0,d_model,2)).float()*-(math.log(10000.0)/d_model)

        pe[:,0::2]=torch.sin(pos*dv_trm)
        pe[:,1::2]=torch.cos(pos*dv_trm)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        return x+self.pe[:,:x.size(1)]