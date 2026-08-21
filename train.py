import torch
from protdiffusion.model import ProtDiffusion
from protdiffusion.data import Protein, ProteinDataset
from protdiffusion.data.tokenizer import ProteinTokenizer
from torch.utils.data import DataLoader
from protdiffusion.geometry import Rigid, Rotation
from protdiffusion.config import device
from protdiffusion.loss import RigidLoss
from torch.optim import Adam


tokenizer = ProteinTokenizer()

##TOO LARGE "1GB1","1AON", "1FAT", "1L2Y"
codes = [
    "1UBQ", "1CRN", "2PTC", "4HHB", "1LYZ", "1AKI", "1MBO", "1HHO", "1TIM", 
    "1GFL", "1CAG", "1R69", "1VII", "2CI2", "1SHG", "1ENH", "1ROP", 
    "1PGB", "1PRB", "1HRC", "1BRS", "2RN2", "1CHO", "1CDT", "1GB1","1AON", "1FAT", "1L2Y"
]
MAX_RESIDUES = 600
codes = [ code for code in codes if Protein.from_codes([code])[0].__len__() <= MAX_RESIDUES ]
codes = codes[:1]

prots = Protein.from_codes(codes)

dataset = ProteinDataset(proteins=prots)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    collate_fn=ProteinDataset.collate_fn
)
model = ProtDiffusion(trunks=10, max_residues=MAX_RESIDUES)
criterion = RigidLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
model.train()
#print(model)
epochs = 20

for epoch in range(epochs):
    for tokens, rigids, mask, max_residues in data_loader:
        B, L = tokens.shape
        T = torch.randint(model.diffuser.num_timesteps, (B, ))
        noise_t, noise_pred = model(tokens=tokens, rigids=rigids, timestep=T, mask=mask)

        loss = criterion(noise_pred, noise_t, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
    print(f"Epoch {epoch+1}: Loss: {loss:.6f}")