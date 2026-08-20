import torch
from protdiffusion.model import RFDiffusion
from protdiffusion.data import Protein, ProteinDataset
from protdiffusion.data.tokenizer import ProteinTokenizer
from torch.utils.data import DataLoader
from protdiffusion.geometry import Rigid, Rotation
from protdiffusion.config import device



tokenizer = ProteinTokenizer()

##TOO LARGE "1GB1","1AON",
codes = [
    "1UBQ", "1CRN", "2PTC", "4HHB", "1LYZ", "1AKI", "1MBO", "1HHO", "1TIM", 
    "1GFL", "1CAG", "1R69", "1VII", "2CI2", "1SHG", "1L2Y", "1ENH", "1ROP", 
    "1PGB", "1PRB", "1HRC", "1BRS",  "1FAT", "2RN2", "1CHO", "1CDT"
]

prots = Protein.from_codes(codes[:10])
#print([{c: p.__len__()} for c, p in zip(codes,prots)])
dataset = ProteinDataset(proteins=prots)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    collate_fn=ProteinDataset.collate_fn
)
model = RFDiffusion(trunks=10, max_residues=600)
model.train()
#print(model)
epochs = 2

for epoch in range(epochs):
    for tokens, rigids, mask, max_residues in data_loader:
        B, L = tokens.shape
        T = torch.randint(model.diffuser.num_timesteps, (B, ))
        xt, noise, noise_pred = model(tokens=tokens, mask=mask, rigids=rigids, timestep=T)

        """
            print("Noise Determinant: ", torch.linalg.det(noise_pred.rotation.matrix))
            orthogonality_error = torch.max(
                torch.abs(noise_pred.rotation.matrix.transpose(-1, -2) @ noise_pred.rotation.matrix - torch.eye(3, device=device))
            )
            print("Orthogonality Error: ", orthogonality_error)
        """
        
    print(f"Epoch {epoch+1}")