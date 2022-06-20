import os
import re
import numpy as np
import pandas as pd
from pathlib import Path


class CreateModelData:
    def __init__(self, domain: str):
        '''
        :param domain: Файл, который категорий домена.
        '''
        self.classes = self.__init_predict(domain)

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def __init_predict(self, path: str):
        '''
        subtopic_true - позволяет производить валидацию.
        :param path: Данные холодного старта.
        :return: Инициализированный набор данных.
        '''
        df = pd.read_csv(self.path(path))
        c = list(
            {i.strip().lower() for i in np.append(df['Тема'], df['Подтема']) if type(i) == str})
        df = pd.DataFrame({'phrase': c, 'subtopic': c, 'true': c})
        df.to_csv(self.path('data/processed/predic.csv'), index=False)
        return c

    def join_train_data(self, full: str, synonyms: str):
        d = pd.read_csv(self.path(synonyms)).merge(pd.read_csv(self.path(full)), how="inner",
                                                   left_on='Synonyms', right_on='item')
        d = d.loc[d.Result == 1, ['item', 'Topic', 'frequency']].rename(
            columns={'item': 'phrase', 'Topic': 'subtopic'}).drop_duplicates('phrase')
        d['subtopic'] = d.subtopic.apply(lambda x: x.strip().lower())
        # Удаление фраз вида: "avonтема"
        d = d.loc[d.subtopic.isin(self.classes)].reset_index(drop=True)
        d.drop([i for i, p in enumerate(d.phrase) if re.compile("[A-z]+").findall(p)], inplace=True)
        d.to_csv('data/processed/train.csv', index=False)
