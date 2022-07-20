import os
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from src.data import CreateModelData
from src.models import Classifier


class ModelTraining:
    run_model = False

    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.train = self.__read_train('data/processed/marked-up-join.csv')
        self.init_df = pd.read_csv('data/processed/init_df.csv')
        self.init_size = self.init_df.shape[0]

    def __read_train(self, train_file: str):
        """
        Загрузка набора данных для снятия метрик.
        :param train_file: Путь до файла.
        :return: Агрегированный набор данных.
        """
        train = pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False)[
            ['phrase', 'subtopic']]
        train['true'] = train['subtopic']
        return train.groupby(by='phrase').agg(subtopic=('subtopic', 'unique'),
                                              true=('true', 'unique')).reset_index()

    # There may be data preprocessing or it may be placed in a separate class
    def __update_predict_df(self, markup: pd.DataFrame):
        '''
        Созраняем размеченные данные в таблицу. Обновляем тренировчный набор.
        :param markup: Разметка полученная разметчиками или моделью.
        '''
        self.init_df = pd.concat([self.init_df, markup], ignore_index=True)
        self.init_df.to_csv(self.path('data/processed/init_df.csv'))

    def __save_metrics(self, df, limit, batch_size, name):
        df.to_csv(self.path(f'models/predicts/{limit}_{batch_size}_{name}.csv'), index=False)

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    # Upgrade to implementation from PyTorch
    def batch(self, batch_size: int) -> pd.DataFrame:
        batch = self.train[:batch_size]  # Получаем разметку и отправляем в размеченный набор данных
        self.train = self.train.drop(index=batch.index).reset_index(drop=True)
        self.__update_predict_df(batch.explode(['subtopic', 'true']))
        return batch

    def start(self, limit: float, batch_size: int, window: int = 3):
        if not self.classifier.start_model_status:  # Если модель пустая - добавляем данные
            group_all_df = self.init_df.groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_all_df['phrase'].values, group_all_df['subtopic'].values)

        people, model = 0, 0
        all_metrics, model_metrics, model_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                index_limit, all_predict = self.classifier.predict(self.train['phrase'], limit)
                model += index_limit.shape[0]
                pred = pd.DataFrame({'phrase': self.train.iloc[index_limit]['phrase'],
                                     'subtopic': all_predict[index_limit] if
                                     index_limit.shape[0] else [],
                                     'true': self.train.iloc[index_limit]['true']})
                model_df = pd.concat([model_df, pred.explode('subtopic').explode('true')],
                                     ignore_index=True)

                # Оцениваем качество модели, если количество предсказанных объектов больше 10
                if index_limit.shape[0] > 10:
                    metrics = self.classifier.metrics(pred['true'].values, pred['subtopic'].values)
                    metrics[['model_from_val', 'model_from_all', 'people_from_val']] = \
                        index_limit.shape[0], model, people
                    model_metrics = pd.concat([model_metrics, metrics])
                    model_metrics.iloc[-1:, :3] = model_metrics.iloc[-window:, :3].agg('mean')
                # self.train = self.__drop_full_from_train(self.train, pred)
                # self.full = self.full.drop(index=index_limit).reset_index(drop=True)
                # self.__update_predict_df(pred.explode('subtopic').explode('true'))

            # Эмуляция разметки данных разметчиками
            batch = self.batch(batch_size=batch_size)
            people += batch.shape[0]

            # Оцениваем качество модели по батчам
            index_limit, all_predict = self.classifier.predict(batch['phrase'].values, limit)
            metrics = self.classifier.metrics(batch['true'].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[
                                                                                   0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            all_metrics.iloc[-1:, :3] = all_metrics.iloc[-window:, :3].agg('mean')
            if people >= 3000:
                self.run_model = True

            # Добавляем новые индексы в модель
            group_all_df = self.init_df[self.init_size:].groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_all_df['phrase'], group_all_df['subtopic'])
            self.init_size = self.init_df.shape[0]  # Обновляем размер набора данных
            print(all_metrics.iloc[-1])

        self.__save_metrics(all_metrics, limit, batch_size, 'all_metrics')
        self.__save_metrics(model_metrics, limit, batch_size, 'model_metrics')
        self.__save_metrics(model_df, limit, batch_size, 'model_data')


if __name__ == '__main__':
    preproc = CreateModelData('data/raw/Decorative/Domain.csv')
    preproc.join_train_data('data/raw/Decorative/Synonyms_test.csv',
                            'data/raw/Decorative/Full_test.csv')
    print('Формирование данных завершено')
    system = ModelTraining(Classifier('models/adaptation/decorative_0_96_1_perfumery-adaptive.bin'))
    t1 = time()
    system.start(limit=0.97, batch_size=500)
    print(time() - t1)
