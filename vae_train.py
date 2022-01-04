
import argparse
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

from vae import VAE
from vae_simple import VAESimple
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_batch(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,_) in enumerate(dataloader):
        X = X.to(device)

        dec, mu, logs2 = model(X)
        kld_loss, recon_loss, loss = model.elbo_loss(dec, X, mu, logs2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current}/{size}]")

    return loss


def test_batch(dataloader, model):
    num_batches = len(dataloader)
    model.eval()
    total_kld_loss = 0
    total_recon_loss = 0
    total_loss = 0

    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)

            dec, mu, logs2 = model(X)
            kld_loss, recon_loss, loss = model.elbo_loss(dec, X, mu, logs2)
            total_kld_loss += kld_loss
            total_recon_loss += recon_loss
            total_loss += loss

    total_kld_loss /= num_batches
    total_recon_loss /= num_batches
    total_loss /= num_batches
    return total_kld_loss, total_recon_loss, total_loss


def train(net, batch_size, lr, train_dataloader, test_dataloader, checkpoint, epochs=16, retrain=False):
    print("Using {} device".format(device))
    net.to(device)

    params = None
    if os.path.exists(checkpoint) and not retrain:
        params = torch.load(checkpoint)
        model_state, optimizer_state = params['model_state_dict'], params['optimizer_state_dict']
        net.load_state_dict(model_state)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=lr)

    min_loss = params['test_loss'] if params is not None else None
    kld_losses = np.zeros((epochs,))
    recon_losses = np.zeros((epochs,))
    total_losses = np.zeros((epochs,))
    print("---------------Beginning training------------------")

    for t in range(epochs):
        print(f"Epoch {t}\n-----------------------")
        train_loss = train_batch(train_dataloader, net, optimizer)
        kld_loss, recon_loss, loss = test_batch(test_dataloader, net)

        kld_losses[t] = kld_loss
        recon_losses[t] = recon_loss
        total_losses[t] = loss
        if min_loss is None or loss < min_loss:
            min_loss = loss
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': loss,
            }, checkpoint)
            print(f'Saved best weights at epoch {t} with loss {loss:>5f}')

    print("Done!")
    return kld_losses, recon_losses, total_losses


def main(args):
    checkpoint = args.model_name
    net = VAESimple(latent_size=args.latent_size, bernoulli_input=args.binarize_mnist)
    config = {'batch_size': 96, "lr": 1e-2}

    train_dataloader, test_dataloader = utils.mnist_data(config['batch_size'], binarize=args.binarize_mnist)

    kld_losses, recon_losses, train_losses = train(net, config['batch_size'], config['lr'], train_dataloader, test_dataloader, checkpoint=checkpoint)
    print(kld_losses)
    print(recon_losses)
    print(train_losses)

    plt.plot(kld_losses)
    plt.plot(recon_losses)
    plt.plot(train_losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", default="vae.pt")
    parser.add_argument("--latent_size", "-l", default=80, type=int)
    parser.add_argument("--binarize_mnist", "-b", action='store_true')
    parser.add_argument("--simple_model", "-s", action='store_true')
    args = parser.parse_args()
    main(args)