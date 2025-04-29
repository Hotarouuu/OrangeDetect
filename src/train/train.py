from torch.utils.data import DataLoader
import torch



def train_model(model, dataloader, eval, criterion, optimizer, device, epochs):

    model.train()

    for epoch in range(epochs):
        print(f'------Época {epoch+1}------')
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                print(f'Loss Atual: {loss.item()}')

        print('\nAvaliando o modelo...')
        model.eval()
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for X_test, y_test in eval:
                X_test, y_test = X_test.to(device), y_test.to(device)
                pred = model(X_test)
                total_loss += criterion(pred, y_test).item()
                predicted = (pred > 0.5).float()
                predicted = predicted.argmax(dim=1)
                correct += (predicted == y_test).sum().item()

        accuracy = correct / len(eval.dataset)
        print(f'Acurácia: {accuracy * 100:.2f}%')
