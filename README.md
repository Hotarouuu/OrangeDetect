# Orange Detect Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license 
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md 
├── data
│   ├── eval           <- Folder for Validation Data
    ├── train          <- Folder for Training Data
    ├── test           <- Folder for Testing Data
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Folder only for storing artifacts from Weight & Biases
│
├── notebooks          <- Jupyter notebooks for prototypes
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py            <- Configuration file for the local packages
│
└── src                       <- Source code for the project.
    │
    ├── __init__.py            <- Makes 'src' a Python package/module
    │
    ├── config.py              <- Stores useful variables and configurations
    │
    ├── dataloader.py          <- Functions to import and process data
    │
    ├── model.py               <- Contains functions to create a custom ResNet model for the project
    │
    ├── predict.py             <- Code to use the model and make predictions
    │
    └── train.py               <- Code to train the model

```

