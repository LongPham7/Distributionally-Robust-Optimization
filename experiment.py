import torch
import torch.nn as nn
from util_model import loadModel, SimpleNeuralNet
from util_MNIST import retrieveMNISTTrainingData
from distributionallyRO import ProjetcedDRO, FrankWolfeDRO

if __name__ == "__main__":
    model = SimpleNeuralNet()
    model = loadModel(model, 'C:\\Users\\famth\\Desktop\\DRO\\models\\SimpleModel.pt')
    loss_criterion = nn.CrossEntropyLoss()
    dro = FrankWolfeDRO(model, loss_criterion, p=2, q=2)
    #dro = ProjetcedDRO(model, loss_criterion)
    dro.train(budget=1, batch_size=128, epochs=1, steps_adv=5)

    """
    data_loader = retrieveMNISTTrainingData(batch_size=1, shuffle=True)
    iterator = iter(data_loader)
    images, labels = iterator.next()
    displayImage(images, labels)
    images_adv, _ = dro.attack(2, (images, labels))
    images_adv.requires_grad = False
    displayImage(images_adv, labels)
    """
    