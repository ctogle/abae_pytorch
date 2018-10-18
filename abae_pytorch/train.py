from torch.nn.functional import normalize
import torch


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


