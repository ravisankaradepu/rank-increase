import torch

from copy import deepcopy
from numpy import linspace
from functools import partial
from itertools import chain, izip_longest
from matplotlib import pyplot as plt
from argparse import ArgumentParser as AP
from torch.utils.data import TensorDataset, DataLoader

p = AP()
p.add_argument('--inp', type=int, required=True, help='Dimensionality of input')
p.add_argument('--otp', type=int, required=True, help='Dimensionality of output')
p.add_argument('--hidden', type=int, nargs='+', required=True, help='Hidden layers (Left to Right)')
p.add_argument('--init_rank', type=int, required=True, help='Rank to begin with')
p.add_argument('--n_iter', type=int, default=100, help='Number of iterations of gradient descent')
p.add_argument('--n_data', type=int, default=1000, help='Number of datapoints in synthetic dataset')
p.add_argument('--device', type=str, default='cpu', help='PyTorch device string <device_name>:<device_id>')
p.add_argument('--graph', action='store_true', help='Toggle for saving a plot')
p.add_argument('--lr', type=float, default=0.001, help='Learning rate for gradient descent')
p.add_argument('--batchsize', type=int, default=-1, help='Batch Size for experimenting. -1 indicates full batch')
p.add_argument('--fontsize', type=int, default=35, help='Font size for graph labels, ticks and legend.')
p.add_argument('--suffix', type=str, default='', help='Add suffix to graph name')
p.add_argument('--nonlinearity', type=str, default=None, help='Add non-linearity to plots')
p = p.parse_args()


def make_full_rank(X):
    """
    Function to get a full rank approximation of a given 2D matrix
    """
    u, s, v = torch.svd(X)
    dx, dy = X.size(0), X.size(1)
    if s.size(0) == min(dx, dy):
        return X
    else:
        deficient = min(dx, dy) - (s > 1e-03).sum()
        s = torch.cat([s, torch.randn(deficient)], axis=0)
        return u.mm(torch.diag(s).mm(v.t()))


def low_rank_init(mdl, rank_val=1):
    """
    Function to get a low rank approximation of the weights in the layer of a neural net.
    To be used with nn.Module.apply()
    """
    if isinstance(mdl, torch.nn.Linear):
        with torch.no_grad():
            tmp = mdl.weight
            u, s, v = torch.svd(tmp)
            s[rank_val:] = 0
            mdl.weight.data = u.mm(torch.diag(s)).mm(v.t())


def add_noise_to_grad(mdl):
    """
    Add noise to the update. To be used with nn.Module.apply() after the normal update
    is done. Noise is normal (N(0, 0.1))
    Now, the noise is added in the following way:
        - Take the normal GD update (W - alpha * gradW) (Performed prior to this op)
        - Perform SVD on the update
        - Get rank `r`
        - Impute a sampled noise in the `r + 1` position of the `s`
          and reconstruct the SVD
    """
    if isinstance(mdl, torch.nn.Linear):
        with torch.no_grad():
            u, s, v = torch.svd(mdl.weight)
            r = (s > 1e-06).sum()
            if r < s.size(0):
                s[r].normal_()  # 0 to r - 1 is rank 'r'
                if s[r].abs() < s[r - 1]:  # to prevent ill-conditioning
                    s[r] = s[r - 1] + torch.normal(torch.tensor(0.), torch.tensor(0.001)).item()
            mdl.weight.data = u.mm(torch.diag(s).mm(v.t()))

def get_weight_product_rank(net):
    """
    Function to get the rank of the product of the matrices.
    """
    with torch.no_grad():
        product = net[-1].weight
        for i in range(len(net._modules) - 2, 0, -1):
            if isinstance(net[i], torch.nn.Linear):
                product = torch.mm(product, net[i].weight)
        _, s, _ = torch.svd(product)
        return (s > 1e-06).sum().item()


# Construct the neural network
layers = [p.inp] + p.hidden + [p.otp]
layer_tuples = [torch.nn.Linear(fanin, fanout, bias=False) for fanin, fanout in zip(layers[:-1], layers[1:])]
non_lin_tuples = []
if p.nonlinearity is not None:
    if p.nonlinearity == 'relu':
        non_lin_tuples = [torch.nn.ReLU(inplace=True) for _ in range(len(p.hidden))]
    elif p.nonlinearity == 'sigmoid':
        non_lin_tuples = [torch.nn.Sigmoid() for _ in range(len(p.hidden))]
    layer_tuples = [x for x in chain(*zip_longest(layer_tuples, non_lin_tuples)) if x is not None]
layer_net = torch.nn.Sequential(*layer_tuples)
del layer_tuples
del non_lin_tuples
print(type(layer_net[0]))

# Random dataset
X = torch.rand(p.n_data, p.inp) * 10 - 5
Y = torch.rand(p.n_data, p.otp) * 2 - 1
X = make_full_rank(X)
Y = make_full_rank(Y)

layer_net = layer_net.to(device=p.device)

# Get low rank weights
layer_net.apply(partial(low_rank_init, rank_val=p.init_rank))

layer_net_noisy = deepcopy(layer_net)
with torch.no_grad():
    for i in range(0, len(layers) - 1):
        if isinstance(layer_net[i], torch.nn.Linear):
            assert (layer_net[i].weight - layer_net_noisy[i].weight).abs().sum().item() < 1e-06, "Deepcopy failed"

layer_net_noisy = layer_net_noisy.to(device=p.device)

# Set optimizers and get initial ranks for weights and their product
optimizer = torch.optim.SGD(layer_net.parameters(), lr=p.lr)
optimizer_noisy = torch.optim.SGD(layer_net_noisy.parameters(), lr=p.lr)
loss = torch.nn.MSELoss()

# Rank list
prod_ranks = []
prod_ranks_noisy = []

# Loss list
losses = []
losses_noisy = []

prod_ranks.append(get_weight_product_rank(layer_net))
prod_ranks_noisy.append(get_weight_product_rank(layer_net_noisy))

tr_dataset = TensorDataset(X, Y)
bs = X.size(0) if p.batchsize == -1 else p.batchsize

tr_loader = DataLoader(tr_dataset, batch_size=bs)

# Training loop
iters = 0
for zzz in range(0, p.n_iter):
    print(zzz)
    for i, (X_, Y_) in enumerate(tr_loader):
        X_, Y_ = X_.to(device=p.device), Y_.to(device=p.device)

        # Normal network
        layer_net.zero_grad()
        cur_loss = loss(layer_net(X_), Y_)
        cur_loss.backward()
        optimizer.step()
        prod_ranks.append(get_weight_product_rank(layer_net))
        losses.append(cur_loss.item())

        # Noisy networks
        layer_net_noisy.zero_grad()
        cur_loss_noisy = loss(layer_net_noisy(X_), Y_)
        cur_loss_noisy.backward()
        optimizer_noisy.step()
        try:
            layer_net_noisy.apply(add_noise_to_grad)
        except RuntimeError as err:
            prod_ranks = prod_ranks[:-1]  # remove previous observation
            losses = losses[:-1]  # remove previous observation
            break
        try:
            prod_ranks_noisy.append(get_weight_product_rank(layer_net_noisy))
        except RuntimeError as err:
            prod_ranks = prod_ranks[:-1]  # remove previous observation
            losses = losses[:-1]  # remove previous observation
            break
        losses_noisy.append(cur_loss_noisy.item())

        iters += 1

print("Sequence of ranks of the product for the normal network:\n{}".format(prod_ranks))
print("Sequence of ranks of the product for the noisy network:\n{}".format(prod_ranks_noisy))

if p.graph:
    # This is the rank section
    max_rank = max([max(prod_ranks_noisy), max(prod_ranks)])
    plt.figure(figsize=(10, 8))
    plt.xlabel("Iterations", fontsize=p.fontsize)
    plt.ylabel("Matrix Rank (product of weights)", fontsize=p.fontsize)
    plt.xlim(-0.5, iters + 0.5)
    plt.ylim(-0.1, max_rank + 0.1)
    plt.xticks(linspace(0, iters, 5))
    plt.tick_params(labelsize=p.fontsize)
    plt.plot(list(range(0, len(prod_ranks))), prod_ranks, 'r', linewidth=4.0, alpha=0.7, label='GD')
    plt.plot(list(range(0, len(prod_ranks_noisy))), prod_ranks_noisy, 'b', linewidth=4.0, alpha=0.7, label='Perturbed GD')
    plt.legend(loc='best', fontsize=p.fontsize)
    plt.tight_layout()
    plt.savefig('Both_product_{}.png'.format(p.suffix), dpi=100)

    # This is the loss section
    max_loss = max([max(losses), max(losses_noisy)])
    plt.figure(figsize=(10, 8))
    plt.xlabel("Iterations", fontsize=p.fontsize)
    plt.ylabel("Loss function value", fontsize=p.fontsize)
    plt.xlim(-0.5, iters)
    plt.xticks(linspace(0, iters, 5))
    plt.tick_params(labelsize=p.fontsize)
    plt.plot(list(range(0, len(losses))), losses, 'r', linewidth=4.0, alpha=0.7, label='GD')
    plt.plot(list(range(0, len(losses_noisy))), losses_noisy, 'b', linewidth=4.0, alpha=0.7, label='Perturbed GD')
    plt.legend(loc='best', fontsize=p.fontsize)
    plt.tight_layout()
    plt.savefig('Loss_comparison_{}.png'.format(p.suffix), dpi=100)

