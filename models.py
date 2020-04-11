import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super(Attention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,query, key, value, mask=None, dropout=None):
        d_k = key.size(-1) # get the size of the key
        scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
        # fill attention weights with 0s where padded
        # if mask is not None: scores = scores.masked_fill(mask, 0)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
          p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value)
        return output

class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = Attention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x

class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        d_feature = int(d_model / n_heads)
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model)

    def forward(self, queries, keys, values, mask=None):
        x = [attn(queries, keys, values, mask=mask)  # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
        # reconcatenate
        x = torch.cat(x, dim=-1)  # (Batch, Seq, D_Feature * n_heads)
        x = self.projection(x)  # (Batch, Seq, D_Model)
        return x

class DBLSTM(nn.Module):
    def __init__(self, args):
        super(DBLSTM, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size,args.hidden_size)
        self.bn1 = nn.BatchNorm1d(args.hidden_size)
        self.bn2 = nn.BatchNorm1d(args.hidden_size)
        self.num_layers = 2
        self.hidden_dim = args.hidden_size
        self.lstm1 = nn.LSTM(args.hidden_size,args.hidden_size_2,bidirectional=True,num_layers=self.num_layers,batch_first=True)
        self.fc3 = nn.Linear(args.hidden_size_2 * 2,args.num_classes)
        if args.self_att =='only_self':
            self.att = Attention()
        if args.self_att == 'heads':
            self.att = MultiHeadAttention(d_model=args.hidden_size_2 * 2,n_heads=1)

    def forward(self, x, mask=None):
        out = F.relu(self.fc1(x))
        # out = self.bn1(out.transpose(2,1))
        out = F.relu(self.fc2(out))
        # out = self.bn2(out.transpose(2,1))
        # out = out.transpose(2,1)
        # out, hidden = self.lstm1(out.view(out.shape[1],out.shape[0],-1))
        out,hidden = self.lstm1(out)
        if self.args.self_att == 'only_self' or self.args.self_att == 'heads':
            out = self.att(out,out,out, mask)

        out = self.fc3(out)
        return out
