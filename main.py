import pandas as pd
from src.data import ClearingPhrases
from src.models import Classifier


class UpdateModel:
    def __init__(self, clearing: ClearingPhrases, classifier: Classifier):
        self.clearing = clearing
        self.classifier = classifier


if __name__ == '__main__':
    full = pd.read_csv('data/input/full.csv')
    phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    parfjum = pd.read_csv('data/input/Parfjum_classifier.csv')
    subtopics = parfjum['Подтема'].dropna().unique()
    model = Classifier()
    model.fit(subtopics=subtopics, n_neighbors=5, weights='distance',
              n_jobs=-1, metric='cosine')
    print(1)
