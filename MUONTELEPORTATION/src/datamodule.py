from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataset(args):
    if args.dataset == "CIFAR10":
        return get_cifar10_dataloaders(args)
    if args.dataset == "MNIST":
        return get_mnist_dataloaders(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_mnist_dataloaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    )


def get_cifar10_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=args.padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=val_transform)
    return (
        DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    )
