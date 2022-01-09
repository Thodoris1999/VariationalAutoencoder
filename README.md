# VariationalAutoencoder
A pytorch variational autoencoder implementation (assignment for Neural Networks & Deep Learning AUTh 9th semester course).
![vaesimple_bernoulli_interpolation](https://github.com/Thodoris1999/VariationalAutoencoder/blob/main/results/vaesimple_bernoulli_interpolation.gif)
![vaesimple_gaussian_interpolation](https://github.com/Thodoris1999/VariationalAutoencoder/blob/main/results/vaesimple_gaussian_interpolation.gif)
## Usage
Create an environment and install dependencies \
`
pip install -r requirements.txt
`

Use `vae_train.py` for training and `vae_eval.py` for inference and introspection of generated images. Both support arguments documented by argparse (`-h` flag).

There are two implementation, a simple CNN (VAESimple, in`vae_simple.py`) and a deeper ResNet (VAE, in`vae.py`). Currently, VAESimple performs better and is used by default. If you want to use another model, change `VAESimple` in `vae_train.py` and `vae_eval.py` to VAE or whatever your desired architecture is.
## Examples
Train model saved to file results/vae_simple.pt with bernoulli modeling of the dataset with latent variable size 60 \
`
python vae_train.py -m results/vae_simple.pt -b -l 60
`

Train model saved to file results/vae_simple.pt with gaussian modeling of the dataset with latent variable size 60 \
`
python vae_train.py -m results/vae_simple.pt -l 60
`

Introspect model with weights from file results/vae_simple.pt and latent size 60 \
`
python vae_eval.py -m results/vae_simple.pt -l 60
`

Introspect model with weights from file results/vae_simple.pt and latent size 60 and binarize generated output \
`
python vae_eval.py -m results/vae_simple.pt -l 60 -b
`
## Pretrained weights
There exist pretrained weights in the results folder
