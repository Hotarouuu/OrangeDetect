import wandb
import torchvision
import torch
from torchvision.models import resnet50
from src.models.resnet_finetuned import create_resnet50_model
from torchvision import transforms
from pathlib import Path
import cv2
from PIL import Image
import os

class OrangeDetect:
    def __init__(self, path: str, artifact_url: str, path_img: str):
        self.artifact_url = artifact_url
        self.path_img = path_img
        self.path = path

    def best_artifact(self):
        artifact_path = os.path.join(self.path,"artifacts")  
        artifact_file = os.path.join(artifact_path, "resnet50-finetuned-v2\\resnet50-finetuned.pth")
        if artifact_file != None:
            print(f"Modelo já baixado em: {artifact_file}")
            return artifact_file

        print("Baixando o modelo...")
        api = wandb.Api()
        artifact = api.artifact(self.artifact_url)
        download_path = artifact.download(artifact_path) 
        return download_path

    def resnet_finetuned(self):
        
        artifact_path = self.best_artifact()
        model, _, _ = create_resnet50_model()

        model.load_state_dict(torch.load(artifact_path, map_location="cpu"))  # weights_only=True não é necessário

        return model, artifact_path
    def pred(self):
        
        model = self.resnet_finetuned()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to tensor first
            transforms.Normalize([0.5], [0.5])  # Then normalize
            ])
        
        img = cv2.imread(self.path_img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img).unsqueeze(0)

        result = model(img_tensor)[0].argmax(dim=0)
        return result
        

