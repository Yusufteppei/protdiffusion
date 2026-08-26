import torch
from protdiffusion.model import ProtDiffusion, ProtNoDiffusion
from protdiffusion.data import Protein, ProteinDataset
from protdiffusion.data.tokenizer import ProteinTokenizer
from torch.utils.data import DataLoader
from protdiffusion.geometry import Rigid, Rotation
from protdiffusion.config import device, ROOT_DIR
from protdiffusion.loss import RigidLoss
from torch.optim import Adam

print(ROOT_DIR)
checkpoint_path = ROOT_DIR / "checkpoints"
print("Training Device: " + device)
tokenizer = ProteinTokenizer()

##TOO LARGE "1GB1","1AON", "1FAT", "1L2Y"
codes = [
    "1UBQ", "1CRN", "2PTC", "4HHB", "1LYZ", "1AKI", "1MBO", "1HHO", "1TIM", 
    "1GFL", "1CAG", "1R69", "1VII", "2CI2", "1SHG", "1ENH", "1ROP", 
    "1PGB", "1PRB", "1HRC", "1BRS", "2RN2", "1CHO", "1CDT", "1GB1","1AON", "1FAT", "1L2Y"
]
MAX_RESIDUES = 600
codes = [ code for code in codes if Protein.from_code(code).__len__() <= MAX_RESIDUES ]
codes = codes[:1]

prots = Protein.from_codes(codes)

dataset = ProteinDataset(proteins=prots)

print(f"Loading Data ... {len(dataset)} proteins, max residues: {MAX_RESIDUES}")
data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    collate_fn=ProteinDataset.collate_fn
)

#model = ProtDiffusion(trunks=10, max_residues=MAX_RESIDUES)
criterion = RigidLoss()
#optimizer = Adam(model.parameters(), lr=1e-4)

epochs = 20


def train_full(epochs=epochs, trunks=10):
    print("Initializing Model ...")
    model = ProtDiffusion(trunks=trunks, max_residues=MAX_RESIDUES)
    try:
        print("Loading Checkpoint ...")
        state_dict = torch.load(f"{checkpoint_path}/{model}.pt")
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Checkpoint not found.")
    except:
        print("Checkpoint load failed.")
        
    optimizer = Adam(model.parameters(), lr=7e-5)
    model.train()
    print("Training ...")
    for epoch in range(epochs):
        for tokens, rigids, mask, _ in data_loader:
            B, L = tokens.shape
            T = torch.randint(model.diffuser.num_timesteps, (B, )) * torch.zeros(B, dtype=torch.int64) 
            noise_t, noise_pred = model(tokens=tokens, rigids=rigids, timestep=T, mask=mask)

            loss = criterion(noise_pred, noise_t, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss: {loss:.4f}")
            torch.save(model.state_dict, f"{checkpoint_path}/{model}.pt")
            print("Checkpoint saved.")
    
    
    return loss, model


def train_no_diffusion(epochs=epochs, trunks=10):
    print("Initializing Model ...")
    model = ProtNoDiffusion(trunks=trunks, max_residues=MAX_RESIDUES)
    model.train()
    optimizer = Adam(model.parameters(), lr=3e-4)
    print("Training ...")
    for epoch in range(epochs):
        for tokens, rigids, mask, _ in data_loader:
            B, L = tokens.shape
            rigids, rigids_pred = model(tokens=tokens, rigids=rigids, mask=mask)

            loss = criterion(rigids_pred, rigids, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss: {loss:.6f}")

            torch.save(model.state_dict, f"{checkpoint_path}/{model}.pt")
            print("Checkpoint Saved.")

    return loss, model

#print([(prot.rigids.translation.mean(), prot.rigids.translation.std()) for prot in prots])

#l1, m1 = train_no_diffusion(5000, 5)
#del(m1)
l2, _ = train_full(11000, 20)

#print(f"No diffusion loss: {l1:.6f}")
print(f"Full diffusion loss: {l2:.6f}")