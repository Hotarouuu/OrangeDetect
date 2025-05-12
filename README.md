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

**There are a few optional steps. If you want to retrain the model and track its training, run the scripts with "optional". If you don't want just run only those that do not have "optional"**


## How to use:


#### Install the requirements:

```python
pip install -r requirements.txt
pip install e . 
```

#### Add the path to the models folder in the .env file:

```
MODELS_FOLDER = '<path>\OrangeDetect\models'
```

#### Execute the streamlit app

```python
streamlit run app.py
```

## How to train your own model:

#### Download the data on Kaggle

Download the file below and place it inside the data folder

https://www.kaggle.com/datasets/mohammedarfathr/orange-fruit-daatset

#### Login on Weights & Biases for tracking (optional):

```python
wandb login
```

#### Download the data on Kaggle and add both data and models path in .env:

```
DATASET_PATH = '<path>\OrangeDetect\data'
MODELS_FOLDER = '<path>\OrangeDetect\models'
```
#### Execute the data processing script:

```
python -m scripts.data_treatment --PATH "path\to\data"
```
#### Execute the training scripts:
You can choose whether you want to track the model or not
```
python -m scripts.training --LEARNING_RATE lr --EPOCHS epochs --NAME experiment name --MODEL resnet50 or resnet101 --TRACKING True or False
```

## Current streamlit UI

Kinda simple but thats it. Im going to improve it further

![image](https://github.com/user-attachments/assets/334d15fc-4acb-401a-bccd-47354f56d27c)

## Update log

**Update 5.2**
- Fixed checkpoint bugs
- Added test metrics to Weights & Biases after training
- Updated ResNet model

**Update 5.1**
- Temporary fix involving the model

**Update 5**

- Added checkpoints during training
- Added the best resnet model with 97% accuracy
