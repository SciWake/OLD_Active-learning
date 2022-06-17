import os
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from src.data import ClearingPhrases
from src.model import Classifier


class ModelTraining:
    run_model = False

    def __init__(self, train_file: str, classifier: Classifier, clearing: ClearingPhrases = None):
        self.clearing = clearing
        self.classifier = classifier
        self.train = self.__read_train(train_file)
        self.init_df = self.__init_data('data/input/parfjum_classifier.csv', 'data/model/in_model.csv')
        self.init_size = self.init_df.shape[0]

    def __init_data(self, path: str, save_path: str) -> pd.DataFrame:
        '''
        subtopic_true - позволяет производить валидацию.
        :param path: Данные холодного старта.
        :param save_path: путь сохранения инициализированного набора данных.
        :return: Инициализированный набор данных.
        '''
        df = pd.read_csv(self.path(path))
        c = list({i.strip().lower() for i in np.append(df['Тема'], df['Подтема']) if type(i) == str})
        df = pd.DataFrame({'phrase': c, 'subtopic': c, 'true': c})
        df.to_csv(self.path(save_path), index=False)
        return df

    def __read_train(self, train_file: str):
        train = pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False)[['phrase', 'subtopic']]
        train['true'] = train['subtopic']
        return train.groupby(by='phrase').agg(subtopic=('subtopic', 'unique'), true=('true', 'unique')).reset_index()

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

    def start(self, limit: float, batch_size: int, window: int = 3):
        if not self.classifier.start_model_status:
            group_init = self.init_df.groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_init['phrase'].values, group_init['subtopic'].values)

        people, model = 0, 0
        all_metrics, marked_metrics, marked_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                index_limit, all_predict = self.classifier.predict(self.train['phrase'], limit)
                model += index_limit.shape[0]
                predict_df = pd.DataFrame({'phrase': self.train.iloc[index_limit].phrase,
                                           'subtopic': all_predict[index_limit] if index_limit.shape[0] else [],
                                           'true': self.train.iloc[index_limit]['true']})
                marked_data = pd.concat([marked_data, predict_df.explode('subtopic').explode('true')],
                                        ignore_index=True)

                # Оцениваем качество модели, если количество предсказанных объектов больше 10
                if index_limit.shape[0] > 10:
                    metrics = self.classifier.metrics(predict_df['true'].values, predict_df['subtopic'].values)
                    metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[0], model, people
                    marked_metrics = pd.concat([marked_metrics, metrics])
                    marked_metrics.iloc[-1:, :3] = marked_metrics.iloc[-window:, :3].agg('mean')

                self.train = self.train.drop(index=index_limit).reset_index(drop=True)
                self.__update_init_df(predict_df.explode('subtopic').explode('true'))

            batch = self.batch(batch_size=batch_size)
            people += batch.shape[0]

            # Оцениваем качество модели по батчам
            index_limit, all_predict = self.classifier.predict(batch['phrase'], limit)
            metrics = self.classifier.metrics(batch['true'].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            all_metrics.iloc[-1:, :3] = all_metrics.iloc[-window:, :3].agg('mean')
            if people >= 3500:
                self.run_model = True

            # Добавляем новые индексы в модель
            group_init = self.init_df[self.init_size:].groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_init['phrase'], group_init['subtopic'])
            self.init_size = self.init_df.shape[0]  # Обновляем размер набора данных
            print(all_metrics.iloc[-1])

        all_metrics.to_csv(self.path(f'data/model/{limit}_{batch_size}_all_metrics.csv'), index=False)
        marked_metrics.to_csv(self.path(f'data/model/{limit}_{batch_size}_marked_metrics.csv'), index=False)
        marked_data.to_csv(self.path(f'data/model/{limit}_{batch_size}_marked.csv'), index=False)


if __name__ == '__main__':
    # full = pd.read_csv('data/input/Parfjum_full_list_to_markup.csv')
    # clearing = ClearingPhrases(full.words_ordered.values)
    # phrases = ClearingPhrases(full.words_ordered.values).get_best_texts
    classifier = Classifier('model/adaptation/bucket.bin', 'model/classifier.pkl')  # 'cache/emb.pkl'
    system = ModelTraining('data/processed/perfumery_train.csv', classifier)
    t1 = time()
    system.start(limit=0.95, batch_size=250)
    print(time() - t1)
