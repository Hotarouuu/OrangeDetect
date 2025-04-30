from torch.utils.data import DataLoader
import torch
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix


class Trainer:
    def __init__(self, model, train_loader, eval_loader, criterion, optimizer, device, epochs):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(f'------ Época {epoch + 1} ------')

            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch % 100 == 0:
                    print(f'Loss Atual: {loss.item():.4f}')

            self.evaluate()

    def evaluate(self):
        print('\nAvaliando o modelo...')
        self.model.eval()
        correct = 0
        total_loss = 0

        with torch.no_grad():
            for X_test, y_test in self.eval_loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                pred = self.model(X_test)
                total_loss += self.criterion(pred, y_test).item()
                predicted = pred.argmax(dim=1)
                correct += (predicted == y_test).sum().item()

        accuracy = correct / len(self.eval_loader.dataset)
        print(f'Acurácia: {accuracy * 100:.2f}%\n')
        self.model.train()  # volta pro modo treino depois de avaliar

    def test(self):

        print('\nTestando o modelo e logando métricas')
        self.model.eval()
        y_correct = []
        y_pred = []
        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                #predicted = (pred > 0.5).float()
                predicted = pred.argmax(dim=1)
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