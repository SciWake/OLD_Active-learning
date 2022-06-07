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
    y = np.array([])
    emb = {}
    t = 0

    def __init__(self, fasttext_model: str, faiss_path: str):
        self.faiss_path = faiss_path
        self.model = fasttext.load_model(str(self.path(fasttext_model)))
        try:
            with open(self.path(faiss_path), 'rb') as f:
                self.index, self.y = pickle.load(f)
                self.start_model_status = 1
        except FileNotFoundError:
            print('No launch model found')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def embeddings(self, texts: list or np.array) -> np.array:
        emb = []
        for text in texts:
            if not self.emb.get(text, np.array([])).shape[0]:
                self.emb[text] = normalize([self.model.get_sentence_vector(text.lower())])[0]
            emb.append(self.emb.get(text))
        return np.array(emb, dtype='float32')

    def add(self, X: np.array, y: np.array):
        if not self.y.shape[0]:
            self.index = faiss.IndexFlat(300)
        self.index.add(self.embeddings(X))
        self.y = np.append(self.y, y)
        with open(self.path(self.faiss_path), 'wb') as f:
            pickle.dump((self.index, self.y), f)
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
            'f1': [f1_score(y_true, y_pred, average='weighted')],
            'precision': [precision_score(y_true, y_pred, average='weighted', zero_division=1)],
            'recall': [recall_score(y_true, y_pred, average='weighted', zero_division=0)],
            'validation_size': [y_true.shape[0]]
        })
