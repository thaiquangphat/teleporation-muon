from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataset(args):
    if(args.dataset == 'CIFAR10'):
        return get_cifar10_dataloaders(args)
    elif(args.dataset == 'CIFAR100'):
        return get_cifar100_dataloaders(args)
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
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=args.padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Validation must be deterministic: no random crop or flipping.
    val_ops = []
    if args.img_size != 32:
        val_ops.append(
            transforms.Resize((args.img_size, args.img_size))
        )
    val_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose(val_ops)

    train_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

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

def get_cifar100_dataloaders(args):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=args.padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Validation must be deterministic: no random crop or flipping.
    val_ops = []
    if args.img_size != 32:
        val_ops.append(
            transforms.Resize((args.img_size, args.img_size))
        )
    val_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose(val_ops)

    train_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

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