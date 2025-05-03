import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

def create_resnet50_model(num_classes=3, lr=0.001, momentum=0.9):
 
    # Import the weights
    weights = ResNet50_Weights.DEFAULT

    # Import pre-trained model
    model = resnet50(weights=weights)

    # freeze the layers -> its necessary for the fine-tuning script
    for param in model.parameters():
        param.requires_grad = False

    # Replaces the FC layer for our classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Defining loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)

    return model, criterion, optimizer

def create_resnet101_model(num_classes=3, lr=0.001, momentum=0.9):
 
    # Import the weights
    weights = ResNet101_Weights.DEFAULT

    # Import pre-trained model
    model = resnet101(weights=weights)

    # freeze the layers -> its necessary for the fine-tuning script
    for param in model.parameters():
        param.requires_grad = False

    # Replaces the FC layer for our classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Defining loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)

    return model, criterion, optimizer