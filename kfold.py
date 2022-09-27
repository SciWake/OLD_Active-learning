import os
import faiss
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from src.models import Classifier
from sklearn.model_selection import KFold
from src.data import CreateModelData


class KFoldClassifier(Classifier):
    def __init__(self, model: str, faiss_path: str = None):
        super(KFoldClassifier, self).__init__(model, faiss_path)

    def add(self, x: np.array, y: np.array):
        """
        Добавление фраз в индекс и сохранение текущего состояния модели.
        Переопределение метода позволяет обнулить список элементов в классификаторе.
        :param x: Набор фраз.
        :param y: Набор категорий.
        :return: self.
        """
        self.y = np.array([])  # Обнуляем список ответов
        self.index = faiss.IndexFlat(300)
        self.index.add(self.embeddings(x))
        self.y = np.append(self.y, y)
        return self


class Stratified:
    def __init__(self, train_file: str, classifier: Classifier):
        self.classifier = classifier
        self.train = self.__read_train(train_file)[:3000]

    def __read_train(self, train_file: str):
        train = pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False)[
            ['phrase', 'subtopic']]
        train['true'] = train['subtopic']
        return train.groupby(by='phrase').agg(subtopic=('subtopic', 'unique'),
                                              true=('true', 'unique')).reset_index()

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def run(self, limit: float, n_splits: int = 3):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        predicts, all_metrics = pd.DataFrame(), pd.DataFrame()

        for train_ind, test_ind in kf.split(self.train):
            train, test = self.train.iloc[train_ind], self.train.iloc[test_ind]

            self.classifier.add(train['phrase'].values, train['subtopic'].values)

            # Снятие метрик
            index_limit, all_predict = self.classifier.predict(test['phrase'].values, limit)
            metrics = self.classifier.metrics(test['true'].iloc[index_limit].values,
                                              all_predict[index_limit])
            all_metrics = pd.concat([all_metrics, metrics], ignore_index=True)
            print(metrics)
            print(len(index_limit), '\n')

            # predict = pd.DataFrame(
            #     {'phrase': test.iloc[index_limit]['phrase'],
            #      'subtopic': all_predict[index_limit],
            #      'true': test.iloc[index_limit]['true']})
            # # Объединение данных
            # predicts = pd.concat([predicts, predict.explode('subtopic').explode('true')],
            #                      ignore_index=True)

        # Сохранение данных
        # all_metrics.to_csv(self.path(f'data/{limit}_all_metrics.csv'), index=False)
        # predicts.to_csv(self.path(f'data/{limit}_predicts.csv'), index=False)
        return all_metrics.agg('mean')['precision']

    def binary_search(self, precision):
        left, right = 0, 1000

        while left <= right:
            mid = (left + right) // 2
            guess = round(self.run(limit=mid / 1000), 2)
            if guess == precision:
                return mid
            if guess > precision:
                right = mid - 1
            else:
                left = mid + 1
        return None

    # def binary_search(self, num):
    #     left, right = 0, 0.99
    #
    #     while left <= right:
    #         mid = (left + right) / 2
    #         guess = self.run(limit=mid)
    #         if guess >= num:
    #             return mid
    #         else:
    #             left = mid + 0.01
    #     return None


if __name__ == '__main__':
    preproc = CreateModelData('data/raw/Parfjum/Domain.csv')
    preproc.join_train_data('data/raw/Parfjum/Synonyms.csv', 'data/raw/Parfjum/Full.csv')
    clas = KFoldClassifier('models/adaptation/apteki_0_67_10_perfume-adaptive.bin')
    system = Stratified('data/processed/marked-up-join.csv', clas)
    t1 = time()
    # system.run(limit=0.98)
    system.binary_search(0.8)
    print(time() - t1)

    # Оцениваем качество модели, если количество предсказанных объектов больше 10
    if limit.shape[0] > 10:
        metrics = self.classifier.metrics(pred['true'].values, pred['subtopic'].values)
        metrics[['model_from_val', 'model_from_all', 'people_from_val']] = \
            limit.shape[0], model, people
        model_metrics = pd.concat([model_metrics, metrics])
        model_metrics.iloc[-1:, :3] = model_metrics.iloc[-window:, :3].agg('mean')