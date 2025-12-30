import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from loss import total_loss
from model import VoxelMorph2D


data_dir = "./data"
checkpoint_path = "voxelmorph2d_mnist.pt"

batch_size = 16
epochs = 5
learning_rate = 1e-3
smoothness_weight = 0.1
seed = 13


class PairMNIST(Dataset):
    def __init__(self, root, train=True):
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        moving, _ = self.dataset[index]
        fixed_index = random.randint(0, len(self.dataset) - 1)
        fixed, _ = self.dataset[fixed_index]
        return moving, fixed


def main():
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairMNIST(root=data_dir, train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = VoxelMorph2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for moving, fixed in loader:
            moving = moving.to(device)
            fixed = fixed.to(device)

            warped, flow = model(moving, fixed)
            loss = total_loss(fixed, warped, flow, smoothness_weight=smoothness_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
