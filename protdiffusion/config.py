import torch

n_res = NUM_RESIDUES = 22
d_res = RESIDUE_DIM = 256
d_pair = PAIR_DIM = 128

n_h = NUM_HEADS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

amino_acids = [
            "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
            "L", "K", "M", "F", "P",
            "S", "T", "W", "Y", "V",
            "X"
]