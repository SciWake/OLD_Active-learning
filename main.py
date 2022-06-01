import os
import pandas as pd
from src.data import ClearingPhrases
from src.models import Classifier


class ModelTraining:
    def __init__(self, classifier: Classifier,
                 clearing: ClearingPhrases = None,
                 train_file: str = 'perfumery_train.csv'):
        self.clearing = clearing
        self.classifier = classifier
        self.train = pd.read_csv(os.path.join(os.getcwd(), 'data', 'processed',
                                              train_file)).sort_values(
            'frequency', ascending=False)
        self.__init_dataset()

    @staticmethod
    def __init_dataset(path: str = 'Parfjum_classifier.csv'):
        df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'input', path))
        df.loc[df['Тема'].isna() & df['Категория'].notna(), 'Тема'] = df.loc[
            df['Тема'].isna() & df['Категория'].notna(), 'Категория']
        df['Тема'].fillna(method='pad', inplace=True)
        df.loc[df['Подтема'].isna(), 'Подтема'] = df.loc[
            df['Подтема'].isna(), 'Тема']
        df['Подтема'].unique()
        df = pd.DataFrame({'phrase': df['Подтема'].unique(),
                           'subtopic': df['Подтема'].unique()})
        df.to_csv(os.path.join(os.getcwd(), 'data', 'model', 'in_model.csv'))

    def __update_init_dataset(self):
        pass

    @property
    def read_trained_data(self):
        return pd.read_csv(
            os.path.join(os.getcwd(), 'data', 'model', 'in_model.csv'))

    def batch(self, batch_size: int = 1000):
        pass

    def train_model(self):
        if classifier.start_model_status:
            pass
        else:
            classifier.fit()

    def start(self):



if __name__ == '__main__':
    full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier()
    system = ModelTraining(classifier, clearing)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    # model.fit(subtopics=subtopics, n_neighbors=5, weights='distance', n_jobs=-1, metric='cosine')
    print(1)
