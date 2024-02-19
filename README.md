
# Text-Based Age and Gender Prediction

## Introduction

This project leverages advanced machine learning techniques to predict the age and gender of individuals based on textual data. Utilizing a combination of natural language processing (NLP) and various classification models, it offers a sophisticated approach to understanding demographic information from text inputs.

## Features

- Text preprocessing and cleaning
- Feature extraction using TF-IDF vectorization
- Age prediction with Support Vector Regression (SVR)
- Gender prediction using XGBoost classifier
- Accuracy and performance metrics evaluation

## Requirements

- Python 3.x
- NumPy
- pandas
- NLTK
- Keras
- LightGBM
- scikit-learn
- XGBoost
- chardet

## Installation

To set up the project environment, follow these steps:

```bash
pip install numpy pandas nltk keras lightgbm scikit-learn xgboost chardet
# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

To use the age and gender prediction models, run the script and input the text for analysis:

```bash
python text_age_gender_prediction.py
```

## How It Works

### Data Preprocessing

- Cleans text data by removing URLs, special characters, and stopwords.
- Lowercases all text to ensure uniformity.

### Feature Extraction

- Utilizes TF-IDF vectorization to transform text data into a format suitable for machine learning models.

### Model Training

- Trains an SVR model for age prediction.
- Trains an XGBoost classifier for gender prediction.

### Prediction

- Takes user input text and outputs predicted age and gender.

## Examples

Input text: "Enter your text here: I love playing football and hanging out with friends."

Output:
```
Predicted Age: 25.4
Predicted Gender: Male
```

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

