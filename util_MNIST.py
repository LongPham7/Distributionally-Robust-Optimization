import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

img_rows, img_cols = 28, 28


def retrieveMNISTTrainingData(batch_size=128, shuffle=True):
    """
    Retrieve a training dataset of MNIST.

    Arguments:
        batch_size: batch size
        shuffle: whether the training data should be shuffled
    Returns:
        data loader for the MNIST training data
    """

    transform = transforms.Compose([transforms.ToTensor()])
    MNIST_train_data = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        MNIST_train_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return train_loader


def retrieveMNISTTestData(batch_size=128, shuffle=False):
    """
    Retrieve a test dataset of MNIST.

    Arguments:
        batch_size: batch size
        shuffle: whether the test data should be shuffled
    Returns:
        data loader for the MNIST test data
    """

    transform = transforms.Compose([transforms.ToTensor()])
    MNIST_test_data = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        MNIST_test_data, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return test_loader


def displayImage(image, label):
    """
    Display an image of a digit from MNIST.

    Arguments:
        image: input image. The shape of this input must be compatible
                with (img_rows, img_cols).
        label: prediction on this input image
    """

    image = image.view((img_rows, img_cols))
    plt.imshow(image, vmin=0.0, vmax=1.0, cmap='gray')
    plt.title("Predicted label: {}".format(label))
    plt.show()


if __name__ == "__main__":
    train_loader = retrieveMNISTTrainingData(batch_size=1, shuffle=False)
    print("MNIST training data are loaded.")
    train_iterator = iter(train_loader)
    images, labels = train_iterator.next()
    print("The type of the image is {}.".format(type(images)))
    print("The size of the image is {}.".format(images.size()))
