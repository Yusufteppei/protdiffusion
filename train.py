import torch
from rfdiffusion.model import RFDiffusion
from rfdiffusion.data import Protein
from rfdiffusion.tokenizer import ResidueTokenizer

tokenizer = ResidueTokenizer()
model = RFDiffusion(trunks=10)

p = Protein.load_pdb()
q = Protein.load_pdb(code="4HHB")


seq = [ p.sequence, q.sequence ]
seq_tokens = tokenizer.batch_encode(seq)


out = model(batch=seq_tokens[0], mask=seq_tokens[1])
print(out)