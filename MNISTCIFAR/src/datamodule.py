from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataset(args):
    if(args.dataset == 'CIFAR10'):
        return get_cifar10_dataloaders(args)
    elif(args.dataset == 'MNIST'):
        return get_mnist_dataloaders(args)

def get_mnist_dataloaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader

def get_cifar10_dataloaders(args):
    transform = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=args.padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader