from .data import preprocess_sentence
from .word2vec import word2vec
from .model import abae
import numpy as np
import torch
import os


class aspect_model:

    def __init__(self, data_path, w2v_path,
                 min_count=10, d_embed=100, n_aspects=10, device='cpu'):
        self.w2v = word2vec(data_path)
        self.w2v.embed(w2v_path, d_embed, min_count=min_count)
        self.w2v.aspect(n_aspects)
        self.ab = abae(self.w2v.E, self.w2v.T).to(device)

    def save_abae(self, abae_path):
        print('saving abae model: "%s"' % abae_path)
        torch.save(self.ab.state_dict(), abae_path)

    def load_abae(self, abae_path):
        print('loading abae model: "%s"' % abae_path)
        self.ab.load_state_dict(torch.load(abae_path))

    def predict(self, *sentences):
        w2i = lambda w: self.w2v.w2i[w] if w in self.w2v.w2i else self.w2v.w2i['<unk>']
        x = [[w2i(w) for w in preprocess_sentence(s)] for s in sentences]
        p_t, z_s = self.ab.predict(torch.LongTensor(x))
        _, i_t = torch.sort(p_t, dim=1)
        return i_t[:, -1]


