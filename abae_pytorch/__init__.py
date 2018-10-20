from .data import preprocess_sentence
from .word2vec import word2vec
from .model import abae


class aspect_model:

    def __init__(self, data_path, w2v_path,
                 min_count=10, d_embed=100, n_aspects=10, device='cpu'):
        self.w2v = word2vec(data_path)
        self.w2v.embed(w2v_path, d_embed, min_count=min_count)
        self.w2v.aspect(n_aspects)
        self.ab = abae(self.w2v.E, self.w2v.T).to(device)

    def predict(self, sentence):
        x = [self.w2v.w2i[w] for w in preprocess_sentence(sentence)]
        return x


        w2i = lambda w: (self.w2v.w2i[w] if w in self.w2v.w2i else self.w2v.w2i['<unk>'])
        for j, (words, length) in enumerate(zip(batch, lengths)):
            index[j, :length] = [w2i(w) for w in words]

        raise NotImplementedError
        return aspect


    def __sample_aspects(self, n=8):
        projection = torch.sort(self.ab.aspects(), dim=1)
        samples = []
        for j, (projs, index) in enumerate(zip(*projection)):
            index = index[-n:].detach().cpu().numpy()
            words = ', '.join([self.w2v.i2w[i] for i in index])
            samples.append(words)
        return samples

    def ___epoch_loss(self, batches, epochsize=100, ortho_reg=0.1):
        losses = []
        with tqdm.tqdm(range(epochsize), total=epochsize, desc='.epoch.') as pbar:
            for b in pbar:
                pos, neg = next(batches)
                r_s, z_s, z_n = self.ab(pos, neg)
                J = self.max_margin_loss(r_s, z_s, z_n).item()
                U = self.orthogonal_regularization(self.ab.T.weight).item()
                loss = (J + ortho_reg * len(pos) * U)
                losses.append(loss)
                x = (b + 1, np.mean(losses))
                pbar.set_description('BATCH: %d | MEAN-LOSS: %0.5f' % x)
        return np.mean(losses)


