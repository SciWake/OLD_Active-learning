import os
import re
import numpy as np
import pandas as pd
from pathlib import Path


class CreateModelData:
    def __init__(self, domain: str):
        """
        Класс позволяет создать данные для обучения классификатора и
        провести владиацию качества модели.
        :param domain: Путь до файла доменной области.
        """
        self.classes = self.__init_predict(domain)

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def __init_predict(self, path: str):
        """
        Создание начального набора данных для решения проблемы холодного старта.
        true - Категория поставленная разметчиком.
        subtopic - Каетогрия прдесказанная моделью.
        :param path: Категории доменной области.
        :return: Уникальные классы второго и третьего уровня, которые содежатся в доменной области.
        """
        d = pd.read_csv(self.path(path))
        c = list({i.strip().lower() for i in np.append(d['Тема'], d['Подтема']) if type(i) == str})
        d = pd.DataFrame({'phrase': c, 'subtopic': c, 'true': c})
        d.to_csv(self.path('data/processed/init_df.csv'), index=False)
        return c

    def __processing(self, d: pd.DataFrame) -> pd.DataFrame:
        """
        Метод преобразует данные к формату, который используется для обучения
        классификатора.
        :param d: Набор данных после операции merge.
        :return: Обработанные данные.
        """
        d['subtopic'] = d.subtopic.apply(lambda x: str(x).strip().lower())
        d = d.loc[d.subtopic.isin(self.classes)].drop_duplicates('phrase', ignore_index=True)
        # Удаление фраз вида: "avonтема"
        return d.drop([i for i, p in enumerate(d.phrase) if re.compile("[A-z]+").findall(p)])

    def join_train_data(self, synonyms: str, full: str, left_on: str = 'Synonyms',
                        right_on: str = 'item'):
        """
        Соеденение размеченных наборов данных в один.
        :param left_on: Левая колонка для операции merge.
        :param right_on: Правая колонка для операции merge.
        :param full: Полный набор данных.
        :param synonyms: Размеченный набор данных.
        """
        d = pd.read_csv(self.path(synonyms)).merge(pd.read_csv(self.path(full)), right_on=right_on,
                                                   left_on=left_on, how="inner")
        d = d.loc[d.Result == 1, ['item', 'Topic', 'frequency']].rename(
            columns={'item': 'phrase', 'Topic': 'subtopic'})
        self.__processing(d).to_csv('data/processed/marked-up-join.csv', index=False)
