import argparse
import numpy as np
import pickle
import os
import pandas as pd
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_preprocess(sentence):
  sentence = sentence.replace('\n', ' ').lower()# lowercase text
  sentence = REPLACE_BY_SPACE_RE.sub(' ', sentence)
  sentence = SYMBOLS_RE.sub('',sentence)# delete symbols text
  sentence = re.sub(r'\d+', '', sentence)
  sentence = ' '.join([w for w in sentence.split() if not w in STOPWORDS])# delete stopwords from text
  return sentence

def csv_to_vector(features):
    feature_vector = vectorizer.transform(features)
    return feature_vector;

if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    #Load data from the location specified by args.train (In this case, an S3 bucket).
    train_features = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")
    train_labels = pd.read_csv(os.path.join(args.train,'train_labels.csv'), index_col=0, engine="python")

    test_features = pd.read_csv(os.path.join(args.test, 'test.csv'), index_col=0, engine="python" )
    
    vectorizer = CountVectorizer()
    vectorizer.fit(train_features)
    
    #Convert training and testing text to a vector 
    train_feature_vector = csv_to_vector(train_features)
    test_feature_vector = csv_to_vector(test_features)
     
    #Train the logistic regression model using the fit method
    model = LogisticRegression().fit(train_X_vector, train_labels)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'image':
        pytesseract.pytesseract.tesseract_cmd = "/usr/share/tesseract-ocr/4.00/tessdata"
        text = []
        img = cv2.imread(request_body)
        text = pytesseract.image_to_string(img)
        text = text_preprocess(text)
        text_vector = csv_to_vector(text)
        return text_vector
    else:
        raise ValueError("This model only supports image input")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction):
    result = prediction.detach().numpy().squeeze()
    return result
