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
        self.train = self.__read_train('data/processed/train.csv')
        self.predict = pd.read_csv('data/processed/predict.csv')
        self.init_size = self.predict.shape[0]

    def __read_train(self, train_file: str):
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
        self.predict = pd.concat([self.predict, markup], ignore_index=True)
        self.predict.to_csv(self.path('data/models/in_model.csv'))

    def __save_metrics(self, df, limit, batch_size, name):
        df.to_csv(self.path(f'data/interim/{limit}_{batch_size}_{name}.csv'), index=False)

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
        if not self.classifier.start_model_status:
            group_init = self.predict.groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_init['phrase'].values, group_init['subtopic'].values)

        people, model = 0, 0
        all_metrics, model_metrics, model_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                index_limit, all_predict = self.classifier.predict(self.train['phrase'], limit)
                model += index_limit.shape[0]
                predict_df = pd.DataFrame({'phrase': self.train.iloc[index_limit].phrase,
                                           'subtopic': all_predict[index_limit] if
                                           index_limit.shape[0] else [],
                                           'true': self.train.iloc[index_limit]['true']})
                model_df = pd.concat(
                    [model_df, predict_df.explode('subtopic').explode('true')],
                    ignore_index=True)

                # Оцениваем качество модели, если количество предсказанных объектов больше 10
                if index_limit.shape[0] > 10:
                    metrics = self.classifier.metrics(predict_df['true'].values,
                                                      predict_df['subtopic'].values)
                    metrics[['model_from_val', 'model_from_all', 'people_from_val']] = \
                        index_limit.shape[0], model, people
                    model_metrics = pd.concat([model_metrics, metrics])
                    model_metrics.iloc[-1:, :3] = model_metrics.iloc[-window:, :3].agg('mean')

                self.train = self.train.drop(index=index_limit).reset_index(drop=True)
                self.__update_predict_df(predict_df.explode('subtopic').explode('true'))

            # Эмуляция разметки данных разметчиками
            batch = self.batch(batch_size=batch_size)
            people += batch.shape[0]

            # Оцениваем качество модели по батчам
            index_limit, all_predict = self.classifier.predict(batch['phrase'], limit)
            metrics = self.classifier.metrics(batch['true'].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[
                                                                                   0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            all_metrics.iloc[-1:, :3] = all_metrics.iloc[-window:, :3].agg('mean')
            if people >= 3500:
                self.run_model = True

            # Добавляем новые индексы в модель
            group_init = self.predict[self.init_size:].groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_init['phrase'], group_init['subtopic'])
            self.init_size = self.predict.shape[0]  # Обновляем размер набора данных
            print(all_metrics.iloc[-1])

        self.__save_metrics(all_metrics, limit, batch_size, 'all_metrics')
        self.__save_metrics(model_metrics, limit, batch_size, 'model_metrics')
        self.__save_metrics(model_df, limit, batch_size, 'model_df')


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    CreateModelData('data/raw/parfjum_classifier.csv').join_train_data(
        'data/raw/Parfjum_full_list_to_markup.csv', 'data/raw/Parfjum_CL_Synonyms.csv')
    system = ModelTraining(Classifier('models/adaptation/bucket.bin', 'models/classifier.pkl'))
    t1 = time()
    system.start(limit=0.95, batch_size=250)
    print(time() - t1)
