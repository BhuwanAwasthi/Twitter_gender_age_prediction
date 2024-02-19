

import numpy as np
import pandas as pd
import re
import string
import nltk
import chardet
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

nltk.download('stopwords')
nltk.download('punkt')
with open('/content/Datasheet.csv', 'rb') as f:
      encoding = chardet.detect(f.read())['encoding']
data = pd.read_csv('/content/Datasheet.csv', encoding=encoding)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
                       # Join tokens back into a single string
    cleaned_text = ' '.join(filtered_text)
    return cleaned_text

data['processed_text'] = data['text'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have your processed text in 'data['processed_text']'
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf_age = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()
y_age = data['age'].values

from sklearn.model_selection import train_test_split

X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_tfidf_age, y_age, test_size=0.2, random_state=42)

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# Initialize the SVR model
svr_model = SVR(kernel='linear')  # You can experiment with different kernels like 'rbf', 'poly', etc.

# Fit the model on your dataset
svr_model.fit(X_train_age, y_train_age)

# Predict the ages on the test set
y_pred_age = svr_model.predict(X_test_age)

# Calculate the mean absolute error
mae_age = mean_absolute_error(y_test_age, y_pred_age)
print(f'Mean Absolute Error for age prediction: {mae_age}')
accuracy_threshold = 5  # years
accurate_predictions = np.abs(y_test_age - y_pred_age) <= accuracy_threshold
accuracy_like_metric = np.mean(accurate_predictions)
print(f'Mean Absolute Error for age prediction: {mae_age}')
print(f'Accuracy-like metric for age prediction (within ±{accuracy_threshold} years): {accuracy_like_metric*100:.2f}%')

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Include bi-grams
X_tfidf = tfidf_vectorizer.fit_transform(data['processed_text']).toarray()

# Preparing the labels for gender (binary classification)
le = LabelEncoder()
y_gender = le.fit_transform(data['gender'])

# Initialize the model with the best parameters from your tuning
final_xgb_model = XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Fit the model on your dataset
final_xgb_model.fit(X_tfidf, y_gender)

# Predict the labels for the dataset
y_pred_gender = final_xgb_model.predict(X_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_gender, y_pred_gender)

# Print the final, single accuracy score
print(f'Final Accuracy of the XGBoost model: {accuracy*100:.2f}%')

def predict_age_gender(text):
      # Preprocess the text
          cleaned_text = clean_text(text)  # Use the clean_text function you defined earlier
          transformed_text = tfidf_vectorizer.transform([cleaned_text]).toarray()
          age_prediction = svr_model.predict(transformed_text)[0]  # SVR model for age prediction
          gender_prediction = final_xgb_model.predict(transformed_text)  # XGB model for gender prediction
          predicted_gender = le.inverse_transform(gender_prediction)[0]  # Inverse transform to get the gender label
          return age_prediction, predicted_gender

new_text = input("Enter your text here: ")
predicted_age, predicted_gender = predict_age_gender(new_text)
print(f"Predicted Age: {predicted_age}")
print(f"Predicted Gender: {predicted_gender}")