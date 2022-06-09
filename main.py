import os
import pandas as pd
from time import time
from pathlib import Path
from src.data import ClearingPhrases
from src.models import Classifier


class ModelTraining:
    run_model = False

    def __init__(self, train_file: str, classifier: Classifier, clearing: ClearingPhrases = None):
        self.clearing = clearing
        self.classifier = classifier
        self.train = self.read_train(train_file)
        self.init_df = self.__init_data('data/input/parfjum_classifier.csv', 'data/model/in_model.csv')
        self.init_size = self.init_df.shape[0]

    def __init_data(self, path: str, save_path: str) -> pd.DataFrame:
        '''
        subtopic_true - позволяет производить валидацию.
        :param path: Данные холодного старта.
        :param save_path: путь сохранения инициализированного набора данных.
        :return: Инициализированный набор данных.
        '''
        df = pd.read_csv(self.path(path)).fillna(method="pad", axis=1)['Подтема'].dropna().values
        df = pd.DataFrame({'phrase': df, 'subtopic': df, 'true': df})
        df.to_csv(self.path(save_path), index=False)
        return df

    # There may be data preprocessing or it may be placed in a separate class
    def __update_init_df(self, markup: pd.DataFrame):
        '''
        Созраняем размеченные данные в таблицу. Обновляем тренировчный набор.
        :param markup: Разметка полученная разметчиками или моделью.
        '''
        self.init_df = pd.concat([self.init_df, markup], ignore_index=True)
        self.init_df.to_csv(self.path('data/model/in_model.csv'))

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        batch = self.train[:batch_size]  # Получаем разметку и отправляем в размеченный набор данных
        self.train = self.train.drop(index=batch.index).reset_index(drop=True)
        self.__update_init_df(batch.explode(['subtopic', 'true']))
        return batch

    def read_train(self, train_file: str):
        train = pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False)[['phrase', 'subtopic']]
        train['true'] = train['subtopic']
        return train.groupby(by='phrase').agg(subtopic=('subtopic', 'unique'), true=('true', 'unique')).reset_index()

    def sma(self):
        pass

    def start(self, limit: float, batch_size: int):
        if not self.classifier.start_model_status:
            self.classifier.add(self.init_df['phrase'].values, self.init_df['subtopic'].values)

        people, model = 0, 0
        all_metrics, marked_metrics, marked_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                index_limit, all_predict = self.classifier.predict(self.train['phrase'], limit)
                predict_df = pd.DataFrame({'phrase': self.train.loc[index_limit].phrase,
                                           'subtopic': all_predict[index_limit] if index_limit.shape[0] else [],
                                           'subtopic_true': self.train.loc[index_limit]['subtopic_true']})
                marked_data = pd.concat([marked_data, predict_df.explode('subtopic')], ignore_index=True)
                self.train = self.train.drop(index=index_limit).reset_index(drop=True)
                model += index_limit.shape[0]
                self.__update_init_df(predict_df.explode('subtopic'))

            batch = self.batch(batch_size=batch_size)
            people += batch.shape[0]

            if self.run_model:
                # Оцениваем качество модели на предсказнных ей
                index_limit, all_predict = self.classifier.predict(marked_data['phrase'], limit)
                metrics = self.classifier.metrics(marked_data['subtopic_true'].values, all_predict)
                metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[0], model, people
                marked_metrics = pd.concat([marked_metrics, metrics])

            # Оцениваем качество модели на всех доступных данных
            index_limit, all_predict = self.classifier.predict(batch['phrase'], limit)
            metrics = self.classifier.metrics(batch['true'].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            if metrics['precision'][0] >= 0.98:
                self.run_model = True

            # Добавляем новые индексы в модель
            self.classifier.add(self.init_df['phrase'][self.init_size:], self.init_df['subtopic'][self.init_size:])
            self.init_size = self.init_df.shape[0]  # Обновляем размер набора данных

        all_metrics.to_csv(self.path(f'data/model/{limit}_{batch_size}_all_metrics.csv'), index=False)
        marked_metrics.to_csv(self.path(f'data/model/{limit}_{batch_size}_marked_metrics.csv'), index=False)
        marked_data.to_csv(self.path(f'data/model/{limit}_{batch_size}_marked.csv'), index=False)
        self.classifier.add(self.init_df['phrase'], self.init_df['subtopic'])


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    classifier = Classifier('models/adaptation/best.bin', 'models/classifier.pkl')
    system = ModelTraining('data/processed/perfumery_train.csv', classifier)
    t1 = time()
    system.start(limit=0.80, batch_size=500)
    print(time() - t1)
