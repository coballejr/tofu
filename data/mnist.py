from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def create_training_loader(batch_size: int) -> DataLoader:

    # Load dataset
    dataset = MNIST(root = './data/', train = True, download = True,
                   transform = ToTensor())

    return DataLoader(dataset, batch_size = batch_size, shuffle = True)


def create_testing_loader(batch_size: int) -> DataLoader:

    # Load dataset
    dataset = MNIST(root = './data/', train = False, download = True,
                    transform = ToTensor())

    return DataLoader(dataset, batch_size = batch_size, shuffle = True)

if __name__ == '__main__':
    train_batch_size = 32
    test_batch_size = 1000

    training_loader = create_training_loader(train_batch_size)
    testing_loader = create_testing_loader(test_batch_size)

    for train_idx, (img, label) in enumerate(training_loader):
        img = img.squeeze()
        print('Loaded batch', train_idx)
        print('There are ', img.shape[0], 'imgs of size ', img.shape[1], 'x', img.shape[2])
        print('Pixel values range from ', img.min(), 'to ', img.max())
        print('Labels range from ', label.min(), 'to ', label.max())

    for test_idx, (img, label) in enumerate(testing_loader):
        img = img.squeeze()
        print('Loaded batch', test_idx)
        print('There are ', img.shape[0], 'imgs of size ', img.shape[1], 'x', img.shape[2])
        print('Labels range from ', label.min(), 'to ', label.max())
