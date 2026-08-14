import torch

n_res = NUM_RESIDUES = 22
d_res = RESIDUE_DIM = 256
d_pair = PAIR_DIM = 128

n_h = NUM_HEADS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"