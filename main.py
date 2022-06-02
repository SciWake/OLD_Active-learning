import os
import pandas as pd
from time import time
from sklearn.metrics import classification_report
from pathlib import Path
from src.data import ClearingPhrases
from src.models import Classifier


class ModelTraining:
    def __init__(self, train_file: str, classifier: Classifier, clearing: ClearingPhrases = None):
        self.clearing = clearing
        self.classifier = classifier
        self.train = pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False)[
            ['phrase', 'subtopic']]
        self.__init_df('data/input/parfjum_classifier.csv', 'data/model/in_model.csv')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def __init_df(self, path: str, save_path: str):
        df = pd.read_csv(self.path(path)).fillna(method="pad", axis=1)['Подтема'].dropna().values
        pd.DataFrame({'phrase': df, 'subtopic': df}).to_csv(self.path(save_path), index=False)

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        return self.train[:batch_size]

    def __update_datasets(self, batch: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([pd.read_csv(self.path('data/model/in_model.csv')), batch])
        df.to_csv(self.path('data/model/in_model.csv'))
        self.train.drop(index=batch.index, inplace=True)
        return df

    # There may be data preprocessing or it may be placed in a separate class
    def update_model(self, batch: pd.DataFrame):
        df = self.__update_datasets(batch)
        self.classifier.add(df['phrase'])

    def start(self):
        if not self.classifier.start_model_status:
            df = pd.read_csv(self.path('data/model/in_model.csv'))
            self.classifier.add(df['phrase'])
            self.classifier.start_model_status = 1

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'batch': []}
        while self.train.shape[0]:
            batch = self.batch(batch_size=1000)
            for_training, predict_model = self.classifier.predict(batch['phrase'])
            self.update_model(batch)

            a, p, r = self.classifier.metrics(batch['subtopic'].values, predict_model)
            metrics['accuracy'].append(a)
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['batch'].append(
                batch.shape[0] if len(metrics.get('batch')) == 0 else metrics.get('batch')[-1] +
                                                                      batch.shape[0])


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier('models/adaptation/best.bin', 'models/classifier.pkl')
    system = ModelTraining('data/processed/perfumery_train.csv', classifier)
    t1 = time()
    system.start()
    print(time() - t1)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
