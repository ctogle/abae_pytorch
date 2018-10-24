import torch.nn as nn
import torch
from torch.nn.functional import normalize, softmax


class attention(nn.Module):

    def __init__(self, d_embed):
        super(attention, self).__init__()
        self.M = nn.Linear(d_embed, d_embed)
        self.M.weight.data.uniform_(-0.1, 0.1)

    def forward(self, e_i):
        y_s = torch.mean(e_i, dim=-1)
        d_i = torch.bmm(e_i.transpose(1, 2), self.M(y_s).unsqueeze(2)).tanh()
        a_i = d_i / sum(torch.exp(d_i))
        return a_i.squeeze(1)


class abae(nn.Module):

    def __init__(self, E, T):
        super(abae, self).__init__()
        n_vocab, d_embed = E.shape
        n_aspects, d_embed = T.shape
        self.E = nn.Embedding(n_vocab, d_embed)
        self.T = nn.Embedding(n_aspects, d_embed)
        self.attention = attention(d_embed)
        self.linear = nn.Linear(d_embed, n_aspects)
        self.E.weight = nn.Parameter(torch.from_numpy(E), requires_grad=False)
        self.T.weight = nn.Parameter(torch.from_numpy(T), requires_grad=True)

    def forward(self, pos, negs):
        p_t, z_s = self.predict(pos)
        r_s = normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1)
        e_n = self.E(negs).transpose(-2, -1)
        z_n = normalize(torch.mean(e_n, dim=-1), dim=-1)
        return r_s, z_s, z_n

    def predict(self, x):
        e_i = self.E(x).transpose(1, 2)
        a_i = self.attention(e_i)
        z_s = normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1)
        p_t = softmax(self.linear(z_s), dim=1)
        return p_t, z_s

    def aspects(self):
        E_n = normalize(self.E.weight, dim=1)
        T_n = normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


