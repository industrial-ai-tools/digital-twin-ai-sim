import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import FNO2d


class DarcyDataset(Dataset):
    def __init__(self, coeffs, sols):
        self.coeffs = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(1)
        self.sols = torch.tensor(sols, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx], self.sols[idx]


def train():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    coeffs = np.load("data/coeffs.npy")
    sols = np.load("data/sols.npy")

    n_train = int(len(coeffs) * 0.8)
    train_ds = DarcyDataset(coeffs[:n_train], sols[:n_train])
    val_ds = DarcyDataset(coeffs[n_train:], sols[n_train:])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Model
    model = FNO2d(modes1=12, modes2=12, width=32, n_blocks=4).to(device)
    print(f"Param count: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    os.makedirs("checkpoint", exist_ok=True)
    best_val = float("inf")

    for epoch in range(150):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train L2: {train_loss:.6e} | Val L2: {val_loss:.6e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoint/fno_best.pt")

    print(f"Done. Best val L2: {best_val:.6e}")


if __name__ == "__main__":
    train()