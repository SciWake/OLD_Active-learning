import os
import pickle
import fasttext
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Add a logger
class PredictError(Exception):
    def __init__(self, text):
        self.txt = text


class Classifier:
    start_model_status = 0

    def __init__(self, fasttext_model: str = 'best.bin',
                 classifier_path: str = 'classifier.pkl'):
        """
        :param fasttext_model: Embedding model
        :param classifier_path: KNeighborsClassifier
        """
        self.classifier_path = classifier_path
        self.model = fasttext.load_model(
            os.path.join(os.getcwd(), "models", "adaptation", fasttext_model))
        try:
            with open(os.path.join(os.getcwd(), "models", classifier_path),
                      'rb') as f:
                self.classifier = pickle.load(f)
                self.start_model_status = 1
        except FileNotFoundError:
            print('No launch model found')

    @staticmethod  # The processing should be placed in a separate class
    def _data_preprocessing(text: str) -> str:
        return text.lower()

    def get_embeddings(self, texts: list or np.array) -> np.array:
        vectors = np.array(
            [self.model.get_sentence_vector(self._data_preprocessing(text)) for
             text in texts if type(text) == str])
        return vectors

    def fit(self, subtopics: np.array, phrases: np.array, **kwargs):
        self.classifier = KNeighborsClassifier(**kwargs)
        self.classifier.fit(self.get_embeddings(phrases), subtopics)
        with open(os.path.join(os.getcwd(), "models", self.classifier_path),
                  'wb') as f:
            pickle.dump(self.classifier, f)
        return self

    @staticmethod  # Implement via sorting using argmax
    def allmin(a: np.array, limit: int = 0.95) -> dict:
        if len(a) == 0:
            raise PredictError(f'No objects found')
        all_ = {}
        for i in range(len(a)):
            if a[i] <= limit and a[i] != 0:
                all_[a[i]] = i
        return all_

    @staticmethod  # Implement via sorting using argmax
    def allmax(a: np.array, limit: int = 0.95) -> dict or None:
        if len(a) == 0:
            return None
        all_ = {}
        for i in range(len(a)):
            if a[i] >= limit:
                all_[a[i]] = i
        return all_

    def predict_proba(self, data: np.array) -> list:
        objects = []
        for item in self.classifier.predict_proba(data):
            item_max = self.allmax(item)
            if item_max:
                objects.append(item)
            else:
                objects.append(self.allmin(item))
        return objects

    @property
    def classes_(self):
        return self.classifier.classes_
