import os
import pickle
import fasttext
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score


# Add a logger
class PredictError(Exception):
    def __init__(self, text):
        self.txt = text


class Classifier:
    start_model_status = 0

    def __init__(self, fasttext_model: str = 'best.bin', classifier_path: str = 'classifier.pkl'):
        """
        :param fasttext_model: Embedding model
        :param classifier_path: KNeighborsClassifier
        """
        self.classifier_path = classifier_path
        self.model = fasttext.load_model(
            os.path.join(os.getcwd(), "models", "adaptation", fasttext_model))
        try:
            with open(os.path.join(os.getcwd(), "models", classifier_path), 'rb') as f:
                self.classifier = pickle.load(f)
                self.start_model_status = 1
        except FileNotFoundError:
            print('No launch model found')

    @staticmethod  # The processing should be placed in a separate class
    def _data_preprocessing(text: str) -> str:
        return text.lower()

    def get_embeddings(self, texts: list or np.array) -> np.array:
        return np.array([self.model.get_sentence_vector(self._data_preprocessing(text)) for
                         text in texts if type(text) == str])

    def fit(self, phrases: np.array, subtopics: np.array, **kwargs):
        self.classifier = KNeighborsClassifier(**kwargs)
        self.classifier.fit(self.get_embeddings(phrases), subtopics)
        with open(os.path.join(os.getcwd(), "models", self.classifier_path), 'wb') as f:
            pickle.dump(self.classifier, f)
        return self

    @staticmethod  # Implement via sorting using argmax
    def allmax(a: np.array, limit: int = 0.98) -> dict or None:
        if len(a) == 0:
            return None
        all_limit = []
        all_ = [0]
        max_ = a[0]
        for i in range(len(a)):
            if a[i] >= limit:
                all_limit.append(i)
            if a[i] > max_:
                all_ = [i]
                max_ = a[i]
            elif a[i] == max_:
                all_.append(i)
        return all_limit, all_

    def predict_proba_(self, x: np.array) -> tuple:
        for_training, predict_model = [], []
        for index, item in enumerate(self.classifier.predict_proba(self.get_embeddings(x))):
            limit_max, max_ = self.allmax(item)
            if not limit_max:  # We save indexes where the model is not sure
                for_training.append(index)
            predict_model.append(self.classifier.classes_[max_[0]])
        return for_training, predict_model

    @property
    def classes_(self):
        return self.classifier.classes_

    @staticmethod
    def metrics(y_true, y_pred) -> tuple:
        a = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, average='macro')
        r = recall_score(y_true, y_pred, average='macro')
        return a, p, r
