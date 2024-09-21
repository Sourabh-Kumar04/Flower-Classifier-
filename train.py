# Imports  here
# %matplotlin inline
# %config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torch.nn as nn
import torch.optim  as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import requests
import argparse
import json
from PIL import Image
from collections import OrderedDict

def data_initialize(data):  
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485,0.456,0.406],
                                                                     [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data


def initialize_model(arch, hidden_units, lr, dropout, device):    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new feed-forward network
    class Classifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

            return nn.LogSoftmax(dim=1)(x)

    # Update the model classifier
    model.classifier = Classifier(25088, 4096, 102)

    # Define the creerion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Move the model to device (CPU or GPU)
    model.to(device)
    
    return model, criterion, optimizer


# parser
parser = argparse.ArgumentParser(description='Training for image classifier')
parser.add_argument('data_dir', default='flower', action='store')
parser.add_argument('--save_dir', default='./checkpoint.pth', action='store')
parser.add_argument('--arch', default='vgg16', action='store')
parser.add_argument('--epochs', type=int, default='3', action='store')
parser.add_argument('--print_every', type=int, default='20', action='store')
parser.add_argument('--lr', type=float, default=1e-3, action='store')
parser.add_argument('--dropout',type=float, default=0.3,action='store')
parser.add_argument('--hidden_layers',type=int, default=256,action='store')
parser.add_argument('--gpu', default='False', action='store')

input_args = parser.parse_args()
data_path = input_args.data_dir
checkpoint = input_args.save_dir
arch = input_args.arch
epochs = input_args.epochs
print_every = input_args.print_every
lr = input_args.lr
dropout = input_args.dropout
hidden_layers = input_args.hidden_layers
device = 'cuda' if torch.cuda.is_available() and input_args.gpu else 'cpu'



def main():
    trainloader, validloader, testloader, train_data = data_initialize(data_path)
    model,criterion,optimizer = initialize_model(arch, hidden_layers, lr, dropout, device)

    # Training loop
    epochs = input_args.epochs
    steps = 0
    running_loss = 0
#     print_every = 40

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # clear gradient values
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # find the new weight values
            loss.backward()
            # update the weights
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader)*100:.3f}")
                running_loss = 0
                model.train()
                
    # TODO: Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'input_size': 25088,
        'ouptut_size': 102,
        'epochs':epochs,
        'optimizer_state':optimizer.state_dict(),
        'classifier':model.classifier,
        'model_state':model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')
    print('Training and Saving model sucess!')
          
 
          
if __name__ == "__main__":
          main()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    