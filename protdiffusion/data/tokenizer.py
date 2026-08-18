import torch
from protdiffusion.config import amino_acids


class ProteinTokenizer:
    def __init__(self):
        self.residue_to_id = {aa: i for i, aa in enumerate(amino_acids)}

        self.pad_id = len(self.residue_to_id)

    def encode(self, sequence: str):
        return torch.tensor(
            [
                self.residue_to_id.get(
                    aa.upper(),
                    self.residue_to_id["X"]
                )
                for aa in sequence
            ],
            dtype=torch.long
        )

    def batch_encode(self, sequences):
        tokens = [self.encode(seq) for seq in sequences]

        lengths = torch.tensor(
            [len(x) for x in tokens],
            dtype=torch.long
        )

        max_len = lengths.max().item()

        batch = torch.full(
            (len(tokens), max_len),
            self.pad_id,
            dtype=torch.long
        )

        mask = torch.zeros(
            (len(tokens), max_len),
            dtype=torch.bool
        )

        for i, token in enumerate(tokens):
            L = len(token)
            batch[i, :L] = token
            mask[i, :L] = True

        return batch, mask
