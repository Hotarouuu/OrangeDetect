from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import torch


def testing_model(model, dataloader, device):

    print('\nTestando o modelo e logando métricas')
    model.eval()
    y_correct = []
    y_pred = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            predicted = (pred > 0.5).float()
            predicted = predicted.argmax(dim=1)
            y_pred.append(predicted)
            y_correct.append(y)
    
    y_pred = torch.cat(y_pred, dim=0)
    y_correct = torch.cat(y_correct, dim=0)

    ## Métricas

    f1score = MulticlassF1Score(num_classes=3)
    precision_metric = MulticlassPrecision(num_classes=3)
    recall_metric = MulticlassRecall(num_classes=3)
    confusion_metric = MulticlassConfusionMatrix(num_classes=3)

    f1score.update(y_correct, y_pred)
    precision_metric.update(y_correct, y_pred)
    recall_metric.update(y_correct, y_pred)
    confusion_metric.update(y_correct, y_pred)

    f1sc = f1score.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    confusion = confusion_metric.compute()

    return f1sc, precision, recall, confusion