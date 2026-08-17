from torch.utils.data import DataLoader, Dataset
from rfdiffusion.data import Protein, ProteinTokenizer


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


    def __len__(self):
        return len(self.proteins)