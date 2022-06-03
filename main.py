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
        self.train = self.train.drop(index=markup.index).reset_index(drop=True)

    def start(self, limit: float, batch_size: int):
        if not self.classifier.start_model_status:
            self.classifier.add(self.init_df['phrase'])

        all_metrics, marked_metrics, marked_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            # Размечаем набор данных моделью
            predict_limit, all_predict = self.classifier.predict(self.train['phrase'], limit)
            marked_data = pd.concat([marked_data, self.train.loc[predict_limit]], ignore_index=True)
            self.__update_init_df(self.train.loc[predict_limit])

            # Получаем разметку и отправляем в размеченный набор данных
            batch = self.batch(batch_size=batch_size)
            self.__update_init_df(batch)

            # Оцениваем качество модели на всех доступных данных
            predict_limit, all_predict = self.classifier.predict(self.init_df['phrase'], limit)
            metrics = self.classifier.metrics(self.init_df['subtopic'], self.init_df['subtopic'][all_predict])
            metrics['marked_model'] = predict_limit.shape[0]
            all_metrics = pd.concat([all_metrics, metrics])

            # Оцениваем качество модели на предсказнных ей
            predict_limit, all_predict = self.classifier.predict(marked_data['phrase'], limit)
            metrics = self.classifier.metrics(marked_data['subtopic'], marked_data['subtopic'][all_predict])
            metrics['marked_model'] = predict_limit.shape[0]
            marked_metrics = pd.concat([marked_metrics, metrics])

            # Добавляем новые индексы в модель
            self.classifier.add(self.init_df['phrase'])

        all_metrics.to_csv(f'{limit}_{batch_size}_all.csv', index=False)
        marked_metrics.to_csv(f'{limit}_{batch_size}_marked.csv', index=False)
        marked_data.to_csv(f'{limit}_{batch_size}_marked.csv')


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    classifier = Classifier('models/adaptation/best.bin', 'models/classifier.pkl')
    system = ModelTraining('data/processed/perfumery_train.csv', classifier)
    t1 = time()
    system.start(limit=0.90, batch_size=500)
    print(time() - t1)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
