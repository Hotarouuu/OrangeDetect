# Orange Detect Project

## Description

Projeto criado para detectar doenças Melanose e Cancro Cítrico em laranjas.

## How to use:

### Install the requirements:
    
```python

pip install -r requirements.txt

```

### Login on Weights & Biases:
    
```python
wandb login
```

### Download the data on Kaggle and add the path in .env

```
DATASET_PATH = r'path\OrangeDetect\data'
MODELS_FOLDER = r'path\OrangeDetect\models'
```

### Execute the data processing script:
```

python -m scripts.data_treatment --PATH "path/to/data"

```

### Execute the training scripts:

```
python -m scripts.training --LEARNING_RATE lr --EPOCHS epochs --NAME experiment name --MODEL resnet50 ou resnet101
```
Execute the streamlit app

```python

streamlit run app.py

```

