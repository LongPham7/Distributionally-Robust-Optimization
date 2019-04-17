import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util_MNIST import retrieveMNISTTestData
from util_model import MNISTClassifier, loadModel

"""
This module is for sanity checking. Most of the code in this module is
attributed to a tutorial in the documentation of PyTorch:
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html. 
"""

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
use_cuda = True  # If GPU is available, choose GPU over CPU.

# FGSM attack code


def fgsm_attack(image, epsilon, data_grad):

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0

    # Loop over all examples in test set
    for i, (data, target) in enumerate(test_loader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.detach()

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
                                                             correct, len(test_loader), final_acc))


if __name__ == "__main__":
    # MNIST Test dataset and dataloader declaration
    test_loader = retrieveMNISTTestData(batch_size=1, shuffle=True)

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (
        use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    filepath_relu = "./experiment_models/MNISTClassifier_relu.pt"
    model_relu = MNISTClassifier(activation='relu')
    model_relu = loadModel(model_relu, filepath_relu)
    model_relu.to(device)

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model_relu.eval()

    # Run test for each epsilon
    for eps in epsilons:
        test(model_relu, device, test_loader, eps)
