from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os 


def dataload(path):


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()]) # I removed the normalize part because of the tokenizer from the ViT

    path_test = os.path.join(path, "test")
    path_train = os.path.join(path, "train")
    path_eval = os.path.join(path, "eval")

    test = ImageFolder(root=path_test, transform=transform) # Automatically classifies the folder order as 0, 1 and 2 respectively
    train = ImageFolder(root=path_train, transform=transform)
    eval = ImageFolder(root=path_eval, transform=transform)



    train_dataloader = DataLoader(train, batch_size=32, num_workers=4,shuffle=True)
    test_dataloader = DataLoader(test, batch_size=32, num_workers=4,shuffle=True)
    eval_dataloader = DataLoader(eval, batch_size=32, num_workers=4,shuffle=True)

    return train_dataloader, test_dataloader, eval_dataloader