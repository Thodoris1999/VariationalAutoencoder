
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from vae_simple import VAESimple
import utils
import viz_utils

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    checkpoint = torch.load(args.model_name)
    model = VAESimple(args.latent_size, args.binarize_mnist)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    train_dataloader, test_dataloader = utils.mnist_data(batch_size=args.batch_size)

    imgs = np.zeros((args.sample_size+1, args.batch_size, 28, 28))
    for X,_ in test_dataloader:
        X = X.to(device)
        imgs[0] = X.reshape((args.batch_size,28,28)).cpu().detach().numpy() 
        mu, logs2 = model.encode(X)
        for i in range(args.sample_size):
            imgs[i+1] = model.generate(mu, logs2)
        break

    img_random = model.generate_random(1)
    #plt.imshow(img_random[0], cmap='gray_r')
    viz_utils.viz_img_block(imgs)
    img_rand_interp = model.generate_random_interpolate(15, 20)
    viz_utils.animate_imgs(img_rand_interp)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", default="vae.pt")
    parser.add_argument("--latent_size", "-l", default=80, type=int)
    parser.add_argument("--binarize_mnist", "-b", action='store_true')
    parser.add_argument("--batch_size", default=5, type=int, help="Number of test images to generate reconstructions from")
    parser.add_argument("--sample_size", "-s", default=8, type=int, help="Number of sampled reconstructions per test image")
    args = parser.parse_args()
    main(args)