from torch.utils.data import DataLoader, Dataset
from rfdiffusion.data import Protein, ProteinTokenizer
import torch 
from rfdiffusion.geometry import Rigid


tokenizer = ProteinTokenizer()
class ProteinDataset(Dataset):

    def __init__(self, proteins: list[Protein]):
        self.proteins = proteins


    def __getitem__(self, idx):
        prot = self.proteins[idx]
        #seq = prot.sequence
        tokens = tokenizer.encode(prot.sequence)
        rigids = prot.rigids
        
        return tokens, rigids

    @classmethod
    def collate_fn(cls, batch):
        tokens = [ i[0] for i in batch ]
        protein_rigids = [ i[1] for i in batch ]

        lengths = torch.tensor(
            [len(x) for x in tokens],
            dtype=torch.long
        )

        max_len = lengths.max().item()
        protein_rigids = [ pr._extend(max_len) for pr in protein_rigids ]
        rigids = Rigid.from_list(protein_rigids)


        batch = torch.full(
            (len(tokens), max_len),
            tokenizer.pad_id,
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
            

        return batch, rigids, mask


    def __len__(self):
        return len(self.proteins)