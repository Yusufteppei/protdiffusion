import torch


class ResidueTokenizer:
    def __init__(self):
        self.residue_to_id = {
            "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
            "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
            "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
            "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
            "X": 20,
        }

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