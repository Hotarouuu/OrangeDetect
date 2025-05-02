from torch.utils.data import DataLoader
import torch
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import wandb


class Trainer:
    def __init__(self, model, train_loader, eval_loader, test_dataloader, criterion, optimizer, device, epochs, project="default", config=None, run_name=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.config = config
        self.eval_metrics = None

        wandb.init(project=project, name=run_name, config=config)
        wandb.watch(self.model)

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            print(f'\n------ Epoch {epoch + 1} ------')

            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch % 100 == 0:
                    print(f'Loss: {loss.item():.4f}')
                
                wandb.log({"batch_loss": loss.item(), "epoch": epoch})

            self.evaluate(epoch)

    def evaluate(self, epoch=0):
        print('\nEvaluating...')
        self.model.eval()
        correct = 0
        total_loss = 0
        y_correct = []
        y_pred = []

        metrics = {
            'loss': [],
            'correct': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

        with torch.no_grad():
            for X_test, y_test in self.eval_loader:
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                pred = self.model(X_test)
                loss = self.criterion(pred, y_test).item()
                total_loss += loss

                predicted = pred.argmax(dim=1)
                correct_batch = (predicted == y_test).sum().item()
                correct += correct_batch

                y_pred.append(predicted)
                y_correct.append(y_test)

                f1score = MulticlassF1Score(num_classes=3)
                precision_metric = MulticlassPrecision(num_classes=3)
                recall_metric = MulticlassRecall(num_classes=3)

                f1score.update(y_test, predicted)
                precision_metric.update(y_test, predicted)
                recall_metric.update(y_test, predicted)

                metrics['loss'].append(loss)
                metrics['correct'].append(correct_batch)
                metrics['f1'].append(f1score.compute().tolist())
                metrics['precision'].append(precision_metric.compute().tolist())
                metrics['recall'].append(recall_metric.compute().tolist())

        accuracy = correct / len(self.eval_loader.dataset)
        mean_loss = sum(metrics['loss']) / len(metrics['loss'])

        wandb.log({
            "eval_loss": mean_loss,
            "eval_accuracy": accuracy,
            "eval_f1": sum(metrics['f1']) / len(metrics['f1']),
            "eval_precision": sum(metrics['precision']) / len(metrics['precision']),
            "eval_recall": sum(metrics['recall']) / len(metrics['recall']),
            "epoch": epoch
        })

        print(f'Accuracy: {accuracy * 100:.2f}%\n')

        self.eval_metrics = metrics
        self.model.train()
        return metrics

    def test(self):
        print('\nTesting and logging')
        self.model.eval()
        y_correct = []
        y_pred = []

        with torch.no_grad():
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                predicted = pred.argmax(dim=1)
                y_pred.append(predicted.cpu())
                y_correct.append(y.cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_correct = torch.cat(y_correct, dim=0)

        f1score = MulticlassF1Score(num_classes=3, average="macro")
        precision_metric = MulticlassPrecision(num_classes=3, average="macro")
        recall_metric = MulticlassRecall(num_classes=3, average="macro")
        confusion_metric = MulticlassConfusionMatrix(num_classes=3)

        f1score.update(y_correct, y_pred)
        precision_metric.update(y_correct, y_pred)
        recall_metric.update(y_correct, y_pred)
        confusion_metric.update(y_correct, y_pred)

        f1sc = f1score.compute().item()
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        confusion = confusion_metric.compute().numpy()

        wandb.log({
            "test_f1": f1sc,
            "test_precision": precision,
            "test_recall": recall,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=y_pred.numpy(),
                y_true=y_correct.numpy(),
                class_names=[str(i) for i in range(3)]
            )
        })

        # salvar artefato
        artifact = wandb.Artifact("final_test_metrics", type="evaluation")
        #artifact.add_file("confusion_matrix.png")  # opcional: se salvar local
        wandb.log_artifact(artifact)

        return f1sc, precision, recall, confusion


