from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os 


def dataload(path):


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #v2.Grayscale(num_output_channels=1), Resnet espera 3 canais
        transforms.ToTensor()
    ])

    path_test = os.path.join(path, "test")
    path_train = os.path.join(path, "train")
    path_eval = os.path.join(path, "eval")

    test = ImageFolder(root=path_test, transform=transform) # Automaticamente classifica como 0, 1 e 2 respectivamente a ordem da pasta
    train = ImageFolder(root=path_train, transform=transform)
    eval = ImageFolder(root=path_eval, transform=transform)



    train_dataloader = DataLoader(train, batch_size=32, num_workers=4,shuffle=True)
    test_dataloader = DataLoader(test, batch_size=32, num_workers=4,shuffle=True)
    eval_dataloader = DataLoader(eval, batch_size=32, num_workers=4,shuffle=True)

    return train_dataloader, test_dataloader, eval_dataloader