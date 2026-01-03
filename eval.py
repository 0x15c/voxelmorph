import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import VoxelMorph2D


data_dir = "./data"
checkpoint_path = "voxelmorph2d_mnist.pt"
output_path = "voxelmorph2d_eval.png"

batch_size = 1
seed = 13


class PairMNIST(Dataset):
    def __init__(self, root, train=False):
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

    dataset = PairMNIST(root=data_dir, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model = VoxelMorph2D().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    moving, fixed = next(iter(loader))
    moving = moving.to(device)
    fixed = fixed.to(device)

    with torch.no_grad():
        warped, flow = model(moving, fixed)

    moving_np = moving[0, 0].cpu().numpy()
    fixed_np = fixed[0, 0].cpu().numpy()
    warped_np = warped[0, 0].cpu().numpy()
    flow_x = flow[0, 0].cpu().numpy()
    flow_y = flow[0, 1].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(moving_np, cmap="gray")
    axes[0].set_title("Moving")
    axes[1].imshow(fixed_np, cmap="gray")
    axes[1].set_title("Fixed")
    axes[2].imshow(warped_np, cmap="gray")
    axes[2].set_title("Warped")
    axes[3].quiver(flow_x[::2, ::2], -flow_y[::2, ::2])
    axes[3].set_title("Flow")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
