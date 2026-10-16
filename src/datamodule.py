import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

def get_dataset(args):
    if(args.dataset == 'CIFAR10'):
        return get_cifar10_dataloaders(args)
    elif(args.dataset == 'CIFAR100'):
        return get_cifar100_dataloaders(args)
    elif(args.dataset == 'MNIST'):
        return get_mnist_dataloaders(args)
    elif(args.dataset == 'IMAGENET'):
        return get_imagenet_dataloaders(args)
    

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

def get_imagenet_dataloaders(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            args.img_size,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    resize_size = int((256 / 224) * args.img_size)
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "val"),
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