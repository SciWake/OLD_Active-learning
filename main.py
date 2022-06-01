import os

import numpy as np
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
            'frequency', ascending=False)[['phrase', 'subtopic']]
        self.__init_df_in_model()

    @staticmethod
    def __init_df_in_model(path: str = 'Parfjum_classifier.csv'):
        df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'input', path))
        df.loc[df['Тема'].isna() & df['Категория'].notna(), 'Тема'] = df.loc[
            df['Тема'].isna() & df['Категория'].notna(), 'Категория']
        df['Тема'].fillna(method='pad', inplace=True)
        df.loc[df['Подтема'].isna(), 'Подтема'] = df.loc[
            df['Подтема'].isna(), 'Тема']
        df['Подтема'].unique()
        df = pd.DataFrame({'phrase': df['Подтема'].unique(),
                           'subtopic': df['Подтема'].unique()})
        df.to_csv(os.path.join(os.getcwd(), 'data', 'model', 'in_model.csv'),
                  index=False)

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        return self.train[:batch_size]

    @property
    def read_trained_data(self):
        return pd.read_csv(
            os.path.join(os.getcwd(), 'data', 'model', 'in_model.csv'))

    def __update_datasets(self, batch: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([self.read_trained_data, batch])
        df.to_csv(os.path.join(os.getcwd(), 'data', 'model', 'in_model.csv'))
        self.train.drop(index=batch.index, inplace=True)
        return df

    # There may be data preprocessing or it may be placed in a separate class
    def update_model(self, batch: pd.DataFrame):
        df = self.__update_datasets(batch)
        classifier.fit(df['phrase'], df['subtopic'], n_neighbors=5,
                       weights='distance', n_jobs=-1, metric='cosine')

    def start(self):
        if not classifier.start_model_status:
            df = self.read_trained_data
            classifier.fit(df['phrase'].values, df['subtopic'].values,
                           n_neighbors=15, weights='distance', n_jobs=-1,
                           metric='cosine')
            classifier.start_model_status = 1

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'batch': []}
        while True:
            batch = self.batch(batch_size=1000)
            self.update_model(batch)
            for_training, predict_model = classifier.predict_proba_(
                batch['phrase'].values)

            a, p, r = self.classifier.metrics(batch['subtopic'].values, predict_model)
            metrics['accuracy'].append(a)
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['batch'].append(0 if len(metrics.get('batch')) == 0 else metrics.get('batch')[-1] + batch.shape[0])


from time import time

if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier()
    system = ModelTraining(classifier)
    t1 = time()
    system.start()
    print(time() - t1)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    # model.fit(subtopics=subtopics, n_neighbors=5, weights='distance', n_jobs=-1, metric='cosine')
    print(1)
