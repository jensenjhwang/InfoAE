import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

root = 'data/'

def get_data_new(dataset, batch_size):
    if dataset == 'MNIST':
        train_set = dsets.MNIST(root+'mnist/', train='train', download=True, transform=transforms.ToTensor())
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
        
        data = next(iter(data_loader))

        mean, std = data[0].mean(), data[0].std()

        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        train_val_data = dsets.MNIST(root+'mnist/', train=True, download=True, transform=transform)
        train_data = torch.utils.data.Subset(train_val_data, list(range(50000)))
        val_data = torch.utils.data.Subset(train_val_data, list(range(50000, 60000)))
        # train_data = train_val_data[:50000]
        # val_data = train_val_data[50000:]
        test_data = dsets.MNIST(root+'mnist/', train=False, download=True, transform=transform)

        train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_load = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
        test_load = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
        return train_load, val_load, test_load

def get_data(batch_size):
    transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor()])

    dataset = dsets.MNIST(root+'/MNIST', train='train', 
        download=False, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    return dataloader

# dataset = next(iter(get_data(100)))
# X_train, y_train = dataset
# print(X_train.shape, y_train.shape)

train, val, test = get_data_new("MNIST", 100)
