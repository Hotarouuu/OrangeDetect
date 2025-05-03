import wandb
import torchvision
import torch
from torchvision.models import resnet50
from src.model import create_resnet50_model
from torchvision import transforms
from pathlib import Path
import cv2
from PIL import Image
import os

class Detect:

    """
    A class to handle model detection tasks, including downloading models 
    from Weight & Biases and managing associated file paths.

    This class contains attributes to store the paths for model files, 
    images, and the URL to access the models from Weight & Biases.

    Attributes:
        path (str): The local file path or directory where models are stored.
        path_img (str): The path to the image associated with the model.

    """
    
    def __init__(self, path: str , path_img: str):
        self.path_img = path_img
        self.path = path

    def resnet_finetuned(self):
        
        model, _, _ = create_resnet50_model()

        raw_path = os.listdir(self.path)
        model_path = os.path.join(self.path, raw_path[0])

        model.load_state_dict(torch.load(model_path, map_location="cpu"))  # weights_only=True não é necessário

        return model
    
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
        

