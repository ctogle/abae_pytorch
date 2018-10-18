from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import torch
import mmap
import tqdm
import traceback
import json
import os
from abae_pytorch.utils import linecount


def preprocess(input_path, output_path):
    with open(input_path, 'r') as in_f, open(output_path, 'w') as out_f:
        lmtzr = WordNetLemmatizer().lemmatize
        stop = stopwords.words('english')
        token = CountVectorizer().build_tokenizer()
        lc = linecount(input_path)
        desc = 'preprocessing "%s"' % input_path
        for j, l in tqdm.tqdm(enumerate(in_f), total=lc, desc=desc):
            tokens = [lmtzr(t) for t in token(l.lower()) if not t in stop]
            n_tokens = len(tokens)
            if len(tokens) > 5 and n_tokens < 100:
                out_l = ' '.join(tokens)
                out_f.write(out_l + '\n')


class dataloader:

    def __init__(self, w2i, path, split=None, seed=0):
        self.w2i = w2i
        self.path = path
        self.meta = './.' + os.path.basename(self.path) + '.meta.json'
        self.split = split if split else {'train': 1.0}
        np.random.seed(seed)

    def __enter__(self):
        self.f = open(self.path, 'rb')
        self.data = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_COPY)
        if os.path.isfile(self.meta):
            self.read_meta()
        else:
            self.offsets = dict((s, []) for s in self.split)
            splits, probs = zip(*list(self.split.items()))
            desc = 'finding offsets in "%s"' % self.path
            i = 0
            for j, char in enumerate(tqdm.tqdm(self.data, desc=desc)):
                if char == b'\n':
                    split = splits[np.random.choice(len(probs), p=probs)]
                    self.offsets[split].append((i, j))
                    i = j + 1
            self.linecounts = dict((s, len(self.offsets[s])) for s in self.split)
            self.linecount = sum(self.linecounts[s] for s in self.split)
            self.write_meta()
        return self

    def __exit__(self, *ags):
        if ags[1]:
            traceback.print_exception(*ags)
        self.f.close()
        return True

    def write_meta(self):
        meta = {
            'path': self.path,
            'linecount': self.linecount,
            'linecounts': self.linecounts,
            'offsets': self.offsets,
        }
        with open(self.meta, 'w') as f:
            f.write(json.dumps(meta))

    def read_meta(self):
        with open(self.meta, 'r') as f:
            meta = json.loads(f.read())
        assert(self.path == meta['path'])
        self.linecount = meta['linecount']
        self.linecounts = meta['linecounts']
        self.offsets = meta['offsets']

    def b2i(self, batch):
        batch = [self.data[u:v].decode('utf').split() for u, v in batch]
        lengths = [len(l) for l in batch]
        index = np.zeros((len(batch), max(lengths)))
        w2i = lambda w: (self.w2i[w] if w in self.w2i else self.w2i['<unk>'])
        for j, (words, length) in enumerate(zip(batch, lengths)):
            index[j, :length] = [w2i(w) for w in words]
        return torch.LongTensor(index)

    def batch_generator(self, split='train', device='cpu', batchsize=20, negsize=20):
        linecount = self.linecounts[split]
        batchcount = (linecount // batchsize)
        pos_offsets = self.offsets[split][:]
        neg_offsets = self.offsets[split][:]
        np.random.shuffle(pos_offsets)
        np.random.shuffle(neg_offsets)
        batches = 0
        while True:
            if batches == batchcount:
                np.random.shuffle(pos_offsets)
                np.random.shuffle(neg_offsets)
                batches = 0
            pos_batch = pos_offsets[batches * batchsize:(batches + 1) * batchsize]
            pos_batch = self.b2i(pos_batch)
            neg_batch = np.random.choice(linecount, batchsize * negsize)
            neg_batch = self.b2i([neg_offsets[i] for i in neg_batch])
            batch = (
                pos_batch.to(device),
                neg_batch.to(device).view(batchsize, negsize, -1),
            )
            yield batch
            batches += 1

    #def apply(self, f, split='train', device='cpu', batchsize=100, negsize=20, **kws):
    #    batches = self.batch_generator(split, device, batchsize, negsize)
    #    return f(batches)




