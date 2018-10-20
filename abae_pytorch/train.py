from torch.nn.functional import normalize
import torch.optim as optim
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import numpy as np
import tqdm


def max_margin_loss(r_s, z_s, z_n):
    device = r_s.device
    pos = torch.bmm(z_s.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
    negs = torch.bmm(z_n, r_s.unsqueeze(2)).squeeze()
    J = torch.ones(negs.shape).to(device) - pos.expand(negs.shape) + negs
    return torch.sum(torch.clamp(J, min=0.0))


def orthogonal_regularization(T):
    T_n = normalize(T, dim=1)
    I = torch.eye(T_n.shape[0]).to(T_n.device)
    return torch.norm(T_n.mm(T_n.t()) - I)


def train(ab, dl, device='cuda', epochs=5, epochsize=100,
          initial_lr=0.02, batchsize=100, negsize=20, ortho_reg=0.1):
    batches = dl.batch_generator('train', device, batchsize, negsize)
    i2w = dict((dl.w2i[w], w) for w in dl.w2i)

    opt = optim.Adam(ab.parameters(), lr=initial_lr)
    plot = plotter()

    epoch_losses = collections.defaultdict(list)
    val_loss = validate(ab, dl, device, 'val', epochsize,
                        batchsize, negsize, ortho_reg)
    epoch_losses['Training Loss'].append(float('inf'))
    epoch_losses['Validation Loss'].append(val_loss)
    #ab.sample_aspects(i2w)
    sample_aspects(ab.aspects(), i2w)
    plot(epoch_losses)

    for e in range(epochs):
        train_losses = []
        with tqdm.trange(epochsize) as pbar:
            for b in pbar:
                pos, neg = next(batches)
                r_s, z_s, z_n = ab(pos, neg)
                J = max_margin_loss(r_s, z_s, z_n)
                U = orthogonal_regularization(ab.T.weight)
                loss = J + ortho_reg * batchsize * U
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                x = (e + 1, opt.param_groups[0]['lr'], train_losses[-1])
                d = 'TRAIN EPOCH: %d | LR: %0.5f | MEAN-TRAIN-LOSS: %0.5f' % x
                pbar.set_description(d)

                if b * batchsize % 100 == 0:
                    lr = initial_lr * (1.0 - 1.0 * ((e + 1) * (b + 1)) / (epochs * epochsize))
                    for pg in opt.param_groups:
                        pg['lr'] = lr

        val_loss = validate(ab, dl, device, 'val', epochsize,
                            batchsize, negsize, ortho_reg)
        epoch_losses['Training Loss'].append(np.mean(train_losses))
        epoch_losses['Validation Loss'].append(val_loss)
        #ab.sample_aspects(i2w)
        sample_aspects(ab.aspects(), i2w)
        plot(epoch_losses)


def validate(ab, dl, device='cuda', split='val',
             epochsize=100, batchsize=100, negsize=20, ortho_reg=0.1):
    losses = []
    batches = dl.batch_generator(split, device, batchsize, negsize)
    with tqdm.tqdm(range(epochsize), total=epochsize, desc='validating') as pbar:
        for b in pbar:
            pos, neg = next(batches)
            r_s, z_s, z_n = ab(pos, neg)
            J = max_margin_loss(r_s, z_s, z_n).item()
            U = orthogonal_regularization(ab.T.weight).item()
            losses.append((J + ortho_reg * batchsize * U))
            x = (b + 1, np.mean(losses))
            pbar.set_description('VAL BATCH: %d | MEAN-VAL-LOSS: %0.5f' % x)
    return np.mean(losses)


def plotter(figsize=(8, 4)):
    f, ax = plt.subplots(1, 1, figsize=figsize)

    def plot_losses(losses):
        colors = cm.rainbow(np.linspace(0, 1, len(losses)))
        lines = []
        for loss, color in zip(losses, colors):
            y = losses[loss]
            x = list(range(len(y)))
            l = ax.plot(x, y, color=color, label=loss, lw=4, marker='o')
            lines.append(loss)
        ax.legend(lines)
        ax.semilogy()
        ax.set_title('Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_xticks(x)
        f.canvas.draw()

    return plot_losses


def sample_aspects(projection, i2w, n=8):
    projection = torch.sort(projection, dim=1)
    for j, (projs, index) in enumerate(zip(*projection)):
        index = index[-n:].detach().cpu().numpy()
        words = ', '.join([i2w[i] for i in index])
        print('Aspect %2d: %s' % (j + 1, words))


