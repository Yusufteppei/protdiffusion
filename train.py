import torch
from rfdiffusion.model import RFDiffusion
from rfdiffusion.data import load_pdb
from rfdiffusion.tokenizer import ResidueTokenizer

tokenizer = ResidueTokenizer()
model = RFDiffusion(trunks=10)

p = load_pdb()
q = load_pdb("datasets/1VII.pdb")


seq = [ p.sequence, q.sequence ]
seq_tokens = tokenizer.batch_encode(seq)


out = model(batch=seq_tokens[0], mask=seq_tokens[1])
print(out[0].shape)