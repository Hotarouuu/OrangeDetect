import mlflow
import pandas as pd
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
logged_model = r'file:///C:/Users/Lucas/Documents/GitHub/OrangeDetect/mlruns/862853957408423999/ef9efa9db0674007b7e41fab7eb272b1/artifacts/resnet-50-finetuned'
loaded_model = mlflow.pyfunc.load_model(logged_model)

img = cv2.imread(r'C:\Users\Lucas\Documents\GitHub\OrangeDetect\scripts\c (4610).jpg')
img = cv2.resize(img, (224, 224))

# Convert NumPy array to PIL image
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor first
    transforms.Normalize([0.5], [0.5])  # Then normalize
])

img = transform(img)


# Add batch dimension
img = img.unsqueeze(0)

outputs = loaded_model.predict(img.cpu().numpy())  # Convert to numpy for compatibility with MLflow model

# Predict on a Pandas DataFrame.


