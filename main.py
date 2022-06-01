import os
import pandas as pd
from time import time
from sklearn.metrics import classification_report
from pathlib import Path
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
        self.__init_df('data/input/parfjum_classifier.csv')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def __init_df(self, path: str):
        df = pd.read_csv(self.path(path))
        df = df.fillna(method="ffill", axis=1).dropna(subset=['Подтема'])
        df = pd.DataFrame({'phrase': df['Подтема'].unique(),
                           'subtopic': df['Подтема'].unique()})
        df.to_csv(self.path('data/model/in_model.csv'), index=False)

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        return self.train[:batch_size]

    def __update_datasets(self, batch: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([self.path('data/model/in_model.csv'), batch])
        df.to_csv(self.path('data/model/in_model.csv'))
        self.train.drop(index=batch.index, inplace=True)
        return df

    # There may be data preprocessing or it may be placed in a separate class
    def update_model(self, batch: pd.DataFrame):
        df = self.__update_datasets(batch)
        self.classifier.fit(df['phrase'], df['subtopic'], n_neighbors=5,
                            weights='distance', n_jobs=-1, metric='cosine')

    def start(self):
        if not self.classifier.start_model_status:
            df = pd.read_csv(self.path('data/model/in_model.csv'))
            self.classifier.fit(df['phrase'].values, df['subtopic'].values,
                                n_neighbors=15, weights='distance', n_jobs=-1,
                                metric='cosine')
            self.classifier.start_model_status = 1

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'batch': []}
        while self.train.shape[0]:
            batch = self.batch(batch_size=1000)
            for_training, predict_model = self.classifier.predict_proba_(
                batch['phrase'].values)
            self.update_model(batch)

            a, p, r = self.classifier.metrics(batch['subtopic'].values,
                                              predict_model)
            metrics['accuracy'].append(a)
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['batch'].append(0 if len(metrics.get('batch')) == 0 else metrics.get('batch')[-1] + batch.shape[0])


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier()
    system = ModelTraining(classifier)
    t1 = time()
    system.start()
    print(time() - t1)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
