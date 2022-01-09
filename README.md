# VariationalAutoencoder
A pytorch variational autoencoder implementation (assignment for Neural Networks & Deep Learning AUTh 9th semester course).
## Usage
Create an environment and install dependencies \
`
pip install -r requirements.txt
`

Use `vae_train.py` for training and `vae_eval.py` for inference and introspection of generated images. Both support arguments documented by argparse (`-h` flag).
## Examples
Train model saved to file vae.pt with bernoulli modeling of the dataset with latent variable size 60 \
`
python vae_train.py -m vae.pt -b -l 60
`

Train model saved to file vae.pt with gaussian modeling of the dataset with latent variable size 60 \
`
python vae_train.py -m vae.pt -l 60
`

Introspect model with weights from file vae.pt and latent size 60 \
`
python vae_eval.py -m vae.pt -l 60
`

Introspect model with weights from file vae.pt and latent size 60 and binarize generated output \
`
python vae_eval.py -m vae.pt -l 60 -b
`
