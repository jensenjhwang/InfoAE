import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Directory containing the data.
root = 'data/'

def get_data_new(dataset, batch_size):
    if dataset == 'MNIST':
        train_set = dsets.MNIST(root+'mnist/', train='train', download=True)
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
        
        data = next(iter(data_loader))

        mean, std = data[0].mean(), data[0].std()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        train_val_data = dsets.MNIST(root+'mnist/', train=True, download=True, transform=transform)
        train_data = train_val_data[:50000]
        val_data = train_val_data[50000:]
        test_data = dsets.MNIST(root+'mnist/', train=False, download=True, transform=transform)

        return train_data, val_data, test_data



def get_data(dataset, batch_size):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train', 
                                download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader