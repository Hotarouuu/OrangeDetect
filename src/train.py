from torch.utils.data import DataLoader
import torch
from torch import nn
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
import wandb
import os
from dotenv import load_dotenv
import io

load_dotenv()

model_path = os.getenv("MODELS_FOLDER") 
CHECKPOINT_PATH = os.getenv("CHECKPOINTS_PATH")

best_val_loss = float('inf')


class Trainer:
    def __init__(self, model, train_loader, eval_loader, test_dataloader, criterion, optimizer, scheduler, tokenizer, device, epochs, project="default", config=None, run_name=None, tracking=False):
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
        self.tracking = tracking
        self.run_name = run_name
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        #self.dropout = nn.Dropout(p=0.2)

        if self.tracking:
            wandb.init(project=project, name=run_name, config=config)
            wandb.watch(self.model)

    def train(self,):
        self.model.train()
        global best_val_loss
        for epoch in range(self.epochs):
            print(f'\n------ Epoch {epoch + 1} ------')

            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                X = self.tokenizer(X, return_tensors="pt", do_rescale=False).to(self.device)

                pred = self.model(**X)
                loss = self.criterion(pred['logits'], y)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if batch % 100 == 0:
                    print(f'Loss: {loss.item():.4f}')

                if self.tracking:
                    wandb.log({"batch_loss": loss.item(), "epoch": epoch})
                #X = self.dropout(X)

            _, val_loss = self.evaluate(epoch)
            
            if epoch >= 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), rf"{model_path}\{self.run_name}_bestmodel-finetuned.pth")
                    print(f"Best model saved at epoch {epoch + 1}")
                    print(f"Best model loss: {best_val_loss}")
            
            # Checkpoint
            if self.tracking:
                artifact = wandb.Artifact(f'{self.run_name}_epoch_{epoch + 1}', type='model')
                # Save model state dict to a temporary buffer
                buffer = io.BytesIO()
                torch.save(self.model.state_dict(), buffer)
                buffer.seek(0)
                # Add the buffer to the artifact
                artifact.add_file(buffer, f'checkpoint_{epoch + 1}-finetuned.pth')
                wandb.log_artifact(artifact)
                print(f'Checkpoint {epoch + 1} saved to wandb')
            else:
                # Save checkpoint locally if tracking is disabled
                checkpoint_path = os.path.join(CHECKPOINT_PATH, f'{self.run_name}_epoch_{epoch + 1}-finetuned.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'Checkpoint {epoch + 1} saved to {checkpoint_path}')

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
                X_test = self.tokenizer(X_test, return_tensors="pt", do_rescale=False).to(self.device)
                pred = self.model(**X_test)
                loss = self.criterion(pred['logits'], y_test).item()
                total_loss += loss

                predicted = pred['logits'].argmax(dim=1)
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
        
        if self.tracking:
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
        return metrics, loss

    def test(self, model_state_dict: str):
        print('\nTesting and logging')
        model = self.model
        model.load_state_dict(torch.load(model_state_dict, map_location=self.device))
        model.eval()
        y_correct = []
        y_pred = []

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                X = self.tokenizer(X, return_tensors="pt", do_rescale=False).to(self.device)
                pred = model(**X)
                predicted = pred['logits'].argmax(dim=1)
                y_pred.append(predicted.cpu().flatten())
                y_correct.append(y.cpu().flatten())

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
        # Print results
        print(f"\nTest Results:")
        print(f"F1 Score: {f1sc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{confusion}")


        if self.tracking:
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
            artifact = wandb.Artifact(f"final_test_metrics", type="evaluation")
            metrics_file = os.path.join(model_path, f"test_metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"F1 Score: {f1sc}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Confusion Matrix:\n{confusion}\n")
            artifact.add_file(metrics_file)
            wandb.log_artifact(artifact)

        return f1sc, precision, recall, confusion