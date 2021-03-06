import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util_MNIST import retrieveMNISTTrainingData, retrieveMNISTTestData, displayImage

img_rows, img_cols = 28, 28


class SimpleNeuralNet(nn.Module):
    """
    Simple neural network consisting of one hidden layer for MNIST.
    This neural network is only used as a toy example. 
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 4)
        self.fc1 = nn.Linear(2 * 25 * 25, 10)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = output.view(-1, self.num_flat_features(output))
        output = self.fc1(output)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MNISTClassifier(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def trainModel(model, loss_criterion, optimizer, epochs=25, filepath=None):
    # USe GPU for computation if it is available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Load the neural network on GPU if it is available
    print("The neural network is now loaded on {}.".format(device))

    running_loss = 0.0
    train_loader = retrieveMNISTTrainingData()
    period = 20
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # Load images and labels on a device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            if i % period == period - 1:
                print("Epoch: {}, iteration: {}, loss: {}".format(
                    epoch, i, running_loss / period))
                running_loss = 0.0
    print("Training is complete.")
    if filepath is not None:
        torch.save(model.state_dict(), filepath)
        print("The model is now saved at {}.".format(filepath))


def loadModel(model, filepath):
    """
    Load the set of parameters into the given model.

    Arguments:
        model: a model whose paramters are to be loaded. If model is None,
            the file should contain information about the architecture of
            the model as well as its parameters. 
        filepath: path to the .pt file that stores the parameters (and
            possibly also the neural network's architecutre) to be loaded
    """

    # Load the model on GPU if it is available.
    # Otherwise, use CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = torch.load(filepath)
    else:
        model.load_state_dict(torch.load(filepath, map_location=device))
    return model


def evaluateModelAccuracy(model):
    # Use GPU for computation if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = retrieveMNISTTestData(batch_size=128)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            softmax = nn.Softmax(dim=1)
            _, predicted = torch.max(softmax(outputs).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def evaluateModelSingleInput(model, image):
    # Use GPU for computation if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input = image.view((1, 1, img_rows, img_cols)).to(device)
    otuput = model(input)
    _, prediction = torch.max(otuput.data, 1)
    return prediction.item()


if __name__ == "__main__":

    def experiment(activation, optimizer_type):
        epochs = 25
        # Note that nn.CrosEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
        loss_criterion = nn.CrossEntropyLoss()
        learning_rate = 0.001

        model = MNISTClassifier(activation=activation)
        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("The optimizer type is not recognized.")
        
        # The file paths are only valid in UNIX systems.
        folderpath = "./ERM_models/"
        filename = "MNISTClassifier_{}_{}".format(optimizer_type, activation)

        trainModel(model, loss_criterion, optimizer,
           epochs=epochs, filepath=folderpath + filename)

    experiment("elu", "adam")
    experiment("relu", "adam")
    experiment("elu", "sgd")
    experiment("relu", "sgd")
