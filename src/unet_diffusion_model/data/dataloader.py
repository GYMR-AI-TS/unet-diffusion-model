from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

full_dataset = datasets.CIFAR10("./dataset", train=True, download=True)
train_dataset, val_dataset = random_split(full_dataset, [0.8, 0.2])
test_dataset = datasets.CIFAR10("./dataset", train=False, download=True)

def get_dataloaders():
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
    return train_loader, val_loader, test_loader
