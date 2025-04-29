import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

def create_resnet50_model(num_classes=3, lr=0.001, momentum=0.9):
    """
    Cria o modelo ResNet50 pré-treinado, ajusta a última camada e configura otimizador e loss.

    Args:
        num_classes (int): Número de classes para a saída.
        lr (float): Taxa de aprendizado.
        momentum (float): Momento do otimizador SGD.

    Returns:
        model (torch.nn.Module): Modelo ajustado.
        criterion (torch.nn.Module): Função de perda.
        optimizer (torch.optim.Optimizer): Otimizador.
    """
    # Importar pesos
    weights = ResNet50_Weights.DEFAULT

    # Carregar modelo pré-treinado
    model = resnet50(weights=weights)

    # Congelar todas as camadas
    for param in model.parameters():
        param.requires_grad = False

    # Substituir a última camada (fc) para o número certo de classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Definir loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=momentum)

    return model, criterion, optimizer
