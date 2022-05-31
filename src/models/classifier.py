import os
import fasttext
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Add a logger
class PredictError(Exception):
    def __init__(self, text):
        self.txt = text


class Classifier:
    _classifier = None

    def __init__(self, path_model: str = 'best.bin'):
        """
        :param path_model: Embedding model
        """
        self.model = fasttext.load_model(
            os.path.join(os.getcwd(), "models", "adaptation", path_model))

    @staticmethod  # The processing should be placed in a separate class
    def _data_preprocessing(text: str) -> str:
        return text.lower()

    def get_embeddings(self, texts: list or np.array) -> np.array:
        vectors = np.array(
            [self.model.get_sentence_vector(self._data_preprocessing(text)) for
             text in texts if type(text) == str])
        return vectors

    def fit(self, subtopics: np.array,
            phrases: list or np.array or None = None,
            **kwargs):
        if phrases:
            vectors_phrases = self.get_embeddings(phrases)
        else:
            vectors_phrases = self.get_embeddings(subtopics)

        self._classifier = KNeighborsClassifier(**kwargs)
        self._classifier.fit(vectors_phrases, subtopics)
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
        for item in self._classifier.predict_proba(data):
            item_max = self.allmax(item)
            if item_max:
                objects.append(item)
            else:
                objects.append(self.allmin(item))
        return objects
