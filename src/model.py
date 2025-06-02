import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from transformers import ViTImageProcessor, ViTForImageClassification

def ViT_Model(num_classes=3, lr=3e-4, weight_decay=0.3,):
 
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    tokenizer = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # freeze the layers
    for param in model.parameters():
        param.requires_grad = False

    # Replaces the FC layer for our classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Defining loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,  # learning rate padrão usado no paper
        weight_decay=0.3,  # weight decay para regularização
        betas=(0.9, 0.999)  # valores padrão do Adam
    )

    return model, criterion, optimizer, tokenizer

