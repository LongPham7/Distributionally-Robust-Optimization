import torch
import foolbox
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification
from util_MNIST import retrieveMNISTTestData
from util_model import loadModel, MNISTClassifier

from torchsummary import summary

"""
This module is for sanity check of Foolbox, a Python library for crafting
adversarial exampples. We apply Foolbox's implementation of FGSM on a neural
network trained by empirical risk minimization (ERM). 
"""

def wrapFoolboxModel(model):
    return PyTorchModel(model, (0, 1), num_classes=10, channel_axis=1)

def adversarialAccuracy(model):
    # Use GPU for computation if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("The model is now loaded on {}.".format(device))
    
    pytorch_model = wrapFoolboxModel(model)
    
    # get source image and label
    batch_size = 1
    test_loader = retrieveMNISTTestData(batch_size=batch_size)
    criterion = Misclassification()

    wrong, total = 0, 0
    period = 500
    max_epsilon = 1.0
    epsilons = 5
    for i, (images, labels) in enumerate(test_loader):
        if i == 10000:
            break
        image, label = images[0].numpy(), labels[0].numpy()

        attack = foolbox.attacks.FGSM(pytorch_model, criterion)
        images_adv = attack(image, label, epsilons=epsilons, max_epsilon=max_epsilon)

        total += 1
        if images_adv is not None:
            wrong += 1
        if i%period == period - 1:
            print("Cumulative adversarial attack success rate: {} / {} = {}".format(wrong, total, wrong / total))
    print("Adversarial error rate: {} / {} = {}".format(wrong, total, wrong / total))

if __name__ == "__main__":
    model_relu = MNISTClassifier(activation='relu')
    model_elu = MNISTClassifier(activation='elu')
    
    # These file paths only work on UNIX. 
    filepath_relu = "./experiment_models/MNISTClassifier_relu.pt"
    filepath_elu = "./experiment_models/MNISTClassifier_elu.pt"
    model_relu = loadModel(model_relu, filepath_relu)
    model_elu = loadModel(model_relu, filepath_elu)

    # Display the structure of the neural network
    #summary(model_relu.cuda(), (1, 28, 28))

    print("The result of relu is as follows.")
    adversarialAccuracy(model_relu)
    print("The result of elu is as follows.")
    adversarialAccuracy(model_elu)
