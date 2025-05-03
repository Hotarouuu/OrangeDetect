# Orange Disease Detection using Fine-Tuning and Streamlit Deployment


This project focuses on using fine-tuning techniques to detect Citrus Canker and Melanose in oranges, aiming to provide farmers with a reliable tool for early detection of these diseases. By utilizing pre-trained models and adapting them to the specific features of citrus diseases, the model offers improved accuracy and efficiency.
Key Features:

- Fine-Tuning: Leveraging transfer learning to fine-tune a pre-trained deep learning model specifically for the task of detecting Citrus Canker and Melanose, allowing for faster convergence and higher accuracy with limited data.

- Tracking: Integrated with Weights & Biases (WandB) for monitoring and tracking the model's performance throughout the fine-tuning process. This enables visualization of training metrics, hyperparameter tuning, and performance comparisons.

- Streamlit Deployment: A user-friendly web application built with Streamlit for easy deployment. The app allows users to upload images of oranges, where the model detects the diseases and provides real-time results with a simple and intuitive interface.

- Farmer Assistance: This tool aids farmers in early disease identification, enabling better decision-making and timely interventions to prevent the spread of these diseases and protect crops.
  

## Technologies Used:

- Python (for model training and deployment)

- PyTorch (for fine-tuning the model)

- Streamlit (for interactive web app deployment)

- Weights & Biases (WandB) (for experiment tracking and model monitoring)

## Download the data on Kaggle

https://www.kaggle.com/datasets/mohammedarfathr/orange-fruit-daatset

## How to use:

#### Install the requirements:

```python
pip install -r requirements.txt
pip install e . 
```

#### Login on Weights & Biases for tracking:
    
```python
wandb login
```

#### Download the data on Kaggle and add the path in .env

```
DATASET_PATH = 'path\OrangeDetect\data'
MODELS_FOLDER = 'path\OrangeDetect\models'
```

#### Execute the data processing script:

```
python -m scripts.data_treatment --PATH "path\to\data"
```

#### Execute the training scripts: -> Execute one time ONLY

```
python -m scripts.training --LEARNING_RATE lr --EPOCHS epochs --NAME experiment name --MODEL resnet50 or resnet101
```

#### Execute the streamlit app

```python
streamlit run app.py
```

