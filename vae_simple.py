
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAESimple(nn.Module):
    def __init__(self, latent_size, bernoulli_input):
        super(VAESimple, self).__init__()
        self.latent_size = latent_size
        if bernoulli_input:
            print("Constructing VAESimple with bernoulli reconstruction loss")
            self.recon_loss = F.binary_cross_entropy_with_logits
        else:
            print("Constructing VAESimple with gaussian reconstruction loss")
            self.recon_loss = F.mse_loss
        # input shape [1, 28, 28]
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, [5,5], stride=2, padding=1), # [16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, [3,3], stride=2, padding=1), # [32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, [3,3], stride=2, padding=0), # [64, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3,3)), # [64, 1, 1]
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logs2 = nn.Linear(64, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 1, 1)),
            nn.ConvTranspose2d(64, 64, kernel_size=(3,3)), # [64, 3, 3]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=2, padding=0), # [64, 6, 6]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Upsample(size=(7,7)), # [64, 7, 7] 
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=2, padding=1), # [32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3,3), stride=2, padding=0), # [16, 28, 28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=(28,28)), # [16, 28, 28] 
            nn.ConvTranspose2d(16, 1, (3,3), padding=1) # [1, 28, 28]
        )


    def encode(self, X):
        enc = self.encoder(X)
        mu = self.fc_mu(enc)
        logs2 = self.fc_logs2(enc)
        return mu, logs2


    def generate(self, mu, logs2):
        with torch.no_grad():
            s = torch.exp(0.5*logs2)
            eps = torch.randn_like(mu)
            z = mu+s*eps
            dec = self.decoder(z)
            return dec.reshape(dec.size(0), 28, 28).cpu().detach().numpy()


    def generate_random(self, n):
        with torch.no_grad():
            z = torch.randn((n, self.latent_size), device=device)
            dec = self.decoder(z)
            return dec.reshape(dec.size(0), 28, 28).cpu().detach().numpy()


    # n number of random latent vectors, n_interpolated number of vectors interpolated between each random vector
    # output size (n*n_interpolated,28,28)
    def generate_random_interpolate(self, n, n_interpolated):
        with torch.no_grad():
            z = torch.randn((n+1, self.latent_size), device=device)
            zs = torch.zeros(n*n_interpolated, self.latent_size, device=device)
            for i in range(n):
                k=0
                for l in range(n_interpolated):
                    zs[i*n_interpolated+l] = (1-k)*z[i]+k*z[i+1]
                    print(k)
                    k += 1.0/n_interpolated

            dec = self.decoder(zs)
            return dec.reshape(dec.size(0), 28, 28).cpu().detach().numpy()



    def forward(self, x):
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logs2 = self.fc_logs2(enc)
        s = torch.exp(0.5*logs2)
        # According to paper, sampling just one noise variable is enough for large batch sizes.
        # Would make variable sample size but makes code slower, messier because it affects loss function
        eps = torch.randn_like(mu)
        z = mu+s*eps

        dec = self.decoder(z)
        return dec, mu, logs2

    
    def elbo_loss(self, dec, y, mu, logs2):
        # assume gaussian input data -> max(log likelihood) ~ max(MSE loss)
        reconstruction_loss = self.recon_loss(dec, y)
        # assume bernoulli input data -> max(log likelihood) ~ max(cross entropy loss)
        # reconstruction_loss = F.binary_cross_entropy_with_logits(dec, y)
        kld_loss = torch.mean(-0.5*torch.sum(1+logs2-torch.square(mu)-torch.exp(logs2), dim=1), dim=0)
        kld_weight = 0.005
        total_loss = reconstruction_loss + kld_weight*kld_loss
        return kld_loss, reconstruction_loss, total_loss