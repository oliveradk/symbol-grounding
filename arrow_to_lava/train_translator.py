import os
import argparse
import torch
from torch import optim, nn
from torch.utils.data import random_split, DataLoader

from utils.utils import make_dir
from .arrowlavadataset import ArrowLavaDataset
from .arrowtranslator import ArrowTranlsator


def train_translator(dataset_size=1000, batch_size=4, epochs=100, save_interval=100, param_dir='saved_params',
                     evaluate=True):
    '''
    Trains a convnet to classify the lava location in observations from Arrow Environments
    :param dataset_size:
    :param batch_size:
    :param epochs:
    :param save_interval: how often to save model weights
    :param param_dir: directory to dave model weights to
    :param evaluate: whether to evaluate model after training
    '''

    #load data
    dataset = ArrowLavaDataset(length=int(dataset_size * 1.2), seed=False)
    train_size, test_size = int(dataset_size), int(.2 * dataset_size)
    trainset, testset = random_split(dataset, lengths=[train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    #get model
    model = ArrowTranlsator()

    #define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    DIR_PATH = os.path.join(os.path.dirname(__file__), param_dir)
    make_dir(DIR_PATH)

    def save_model(steps):
        PATH = os.path.join(DIR_PATH, f'translator_net_{steps}.pth')
        torch.save(model.state_dict(), PATH)

    save_model(0)
    steps = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            steps += batch_size
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if steps % save_interval == 0:
                save_model(steps)
        print(f'[{epoch}] loss: {running_loss}')


    print('Finished Training')

    if evaluate:
        evaluate_model(model, testloader)


def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    for data in dataloader:
        images, labels = data
        predicted = model.predict(images)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_interval', type=int)
    parser.add_argument('--param_dir')
    parser.add_argument('--evaluate', type=bool)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    train_translator(**kwargs)

