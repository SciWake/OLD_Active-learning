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
        self.init_df = self.__init_df('data/input/parfjum_classifier.csv',
                                      'data/model/in_model.csv')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def __init_df(self, path: str, save_path: str) -> pd.DataFrame:
        df = pd.read_csv(self.path(path)).fillna(method="pad", axis=1)['Подтема'].dropna().values
        df = pd.DataFrame({'phrase': df, 'subtopic': df})
        df.to_csv(self.path(save_path), index=False)
        return df

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        return self.train[:batch_size]

    # There may be data preprocessing or it may be placed in a separate class
    def __update_init_df(self, markup: pd.DataFrame):
        '''
        Созраняем размеченные данные в таблицу. Обновляем тренировчный набор.
        :param markup: Разметка полученная разметчиками или моделью.
        '''
        self.init_df = pd.concat([self.init_df, markup], ignore_index=True)
        self.init_df.to_csv(self.path('data/model/in_model.csv'))
        self.train = self.train.drop(index=markup.index).reset_index(drop=True)  # Удаление данных размеченных моделью

    def start(self):
        if not self.classifier.start_model_status:
            self.classifier.add(self.init_df['phrase'])
            self.classifier.start_model_status = 1

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'batch': [], 'for_training': []}
        while self.train.shape[0]:
            # Размечаем набор данных моделью
            predict_limit, all_predict = self.classifier.predict(self.train['phrase'], 0.95)
            self.__update_init_df(self.train.loc[predict_limit])  #
            # Получаем разметку и отправляем в размеченный набор данных
            batch = self.batch(batch_size=1000)
            self.__update_init_df(batch)
            # Оцениваем качество модели на всех доступных данных
            _, predict_model = self.classifier.predict(self.init_df['phrase'], 0.95)
            a, p, r = self.classifier.metrics(self.init_df['subtopic'], self.init_df['subtopic'].values[predict_model])
            metrics['accuracy'].append(a)
            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['predict_limit'].append(predict_limit.shape)
            metrics['batch'].append(
                batch.shape[0] if len(metrics.get('batch')) == 0 else metrics.get('batch')[-1] +
                                                                      batch.shape[0])
            # Добавляем новые индексы в модель
            self.classifier.add(self.init_df['phrase'])
        pd.DataFrame(metrics).to_csv('metrics.csv')


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier('models/adaptation/best.bin', 'models/classifier.pkl')
    system = ModelTraining('data/processed/perfumery_train.csv', classifier)
    t1 = time()
    system.start()
    print(time() - t1)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
