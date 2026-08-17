import torch
from rfdiffusion.model import RFDiffusion
from rfdiffusion.data import Protein, ProteinDataset
from rfdiffusion.data.tokenizer import ProteinTokenizer
from torch.utils.data import DataLoader
from rfdiffusion.geometry import Rigid, Rotation



tokenizer = ProteinTokenizer()
model = RFDiffusion(trunks=10)
model.train()
epochs = 20

##TOO LARGE "1GB1","1AON",
codes = [
    "1UBQ", "1CRN", "2PTC", "4HHB", "1LYZ", "1AKI", "1MBO", "1HHO", "1TIM", 
    "1GFL", "1CAG", "1R69", "1VII", "2CI2", "1SHG", "1L2Y", "1ENH", "1ROP", 
    "1PGB", "1PRB", "1HRC", "1BRS",  "1FAT", "2RN2", "1CHO", "1CDT"
]

prots = Protein.from_codes(codes[:10])
#print([{c: p.__len__()} for c, p in zip(codes,prots)])
dataset = ProteinDataset(proteins=prots)

def collate_fn(batch):
    tokens = [ i[0] for i in batch ]
    protein_rigids = [ i[1] for i in batch ]

    lengths = torch.tensor(
        [len(x) for x in tokens],
        dtype=torch.long
    )

    max_len = lengths.max().item()
    protein_rigids = [ pr._extend(max_len) for pr in protein_rigids ]
    rigids = Rigid.from_list(protein_rigids)

    print("Rigid de Finale", rigids.rotation.matrix.shape)

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



data_loader = DataLoader(
    dataset=dataset,
    batch_size=5,
    collate_fn=collate_fn
)


for tokens, rigids, mask in data_loader:
    
    
    out = model(tokens=tokens, mask=mask, rigids=rigids)

print(out)