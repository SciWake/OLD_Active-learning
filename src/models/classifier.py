import os
import pickle
import faiss
import fasttext
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path


# Add a logger
class PredictError(Exception):
    def __init__(self, text):
        self.txt = text


class Classifier:
    start_model_status = 0
    y = None

    def __init__(self, fasttext_model: str, faiss_path: str):
        self.faiss_path = faiss_path
        self.model = fasttext.load_model(str(self.path(fasttext_model)))
        try:
            with open(self.path(faiss_path), 'rb') as f:
                self.index = pickle.load(f)
                self.start_model_status = 1
        except FileNotFoundError:
            print('No launch model found')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def embeddings(self, texts: list or np.array) -> np.array:
        return normalize(np.array([self.model.get_sentence_vector(text.lower()) for text in texts]))

    def add(self, X: np.array, y: np.array):
        self.index = faiss.IndexFlat(300)
        self.index.add(self.embeddings(X))
        self.y = y
        with open(self.path(self.faiss_path), 'wb') as f:
            pickle.dump(self.index, f)
        return self

    def predict(self, x: np.array, limit: float) -> tuple:
        # predict_limit - то, что предсказал модель
        predict_limit, all_predict = [], []
        dis, ind = self.index.search(self.embeddings(x), k=5)
        for i in range(x.shape[0]):
            if any(dis[i] <= 1 - limit):  # We save indexes where the model is not sure
                predict_limit.append(i)
            all_predict.append(self.y[ind[i][0]])
        return np.array(predict_limit), np.array(all_predict)

    @staticmethod
    def metrics(y_true: np.array, y_pred: np.array) -> pd.DataFrame:
        return pd.DataFrame({
            'f1': [f1_score(y_true, y_pred, average='macro')],
            'precision': [precision_score(y_true, y_pred, average='macro', zero_division=1)],
            'recall': [recall_score(y_true, y_pred, average='macro', zero_division=0)],
            'marked_model_size': [0],
            'validation_size': [y_true.shape[0]]
        })
