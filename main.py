import os
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from src.models import Classifier
from src.labelstud.script import LabelStudio
from kfold import Stratified


class ModelTraining:
    run_model, history, limit = False, False, 0.99

    def __init__(self, classifier: Classifier, label: LabelStudio,
                 train_path: str = 'run_data/data.csv', domain_path: str = 'run_data/Domain.csv'):
        self.classifier = classifier
        self.lb = label
        self.train = self.__read_train(train_path)
        self.init_df = self.__init_domain(domain_path)

    def __read_train(self, train_file: str):
        """
        Загрузка набора данных для снятия метрик.
        :param train_file: Путь до файла.
        :return: Агрегированный набор данных.
        """
        return pd.read_csv(self.path(train_file)).sort_values('frequency', ascending=False).dropna(
            subset=['item']).reset_index(drop=True)
        # return train.groupby(by='phrase').agg(subtopic=('subtopic', 'unique'), true=('true', 'unique')).reset_index()

    def __init_domain(self, path: str):
        """
        Создание начального набора данных для решения проблемы холодного старта.
        true - Категория поставленная разметчиком.
        subtopic - Каетогрия прдесказанная моделью.
        :param path: Категории доменной области.
        :return: Уникальные классы второго и третьего уровня, которые содежатся в доменной области.
        """
        d = pd.read_csv(self.path(path))
        c = list({i.strip().lower() for i in np.append(d['Тема'], d['Подтема']) if type(i) == str})
        d = pd.DataFrame({'phrase': c, 'subtopic': c})
        d.to_csv(self.path('data/processed/init_df.csv'), index=False)
        print(f"Сохранение плоского классификатора {self.path('data/processed/init_df.csv')} ")
        return d

    # # There may be data preprocessing or it may be placed in a separate class
    # def __update_predict_df(self, markup: pd.DataFrame):
    #     '''
    #     Созраняем размеченные данные в таблицу. Обновляем тренировчный набор.
    #     :param markup: Разметка полученная разметчиками или моделью.
    #     '''
    #     self.init_df = pd.concat([self.init_df, markup], ignore_index=True)
    #     self.init_df.to_csv(self.path('data/processed/init_df.csv'))

    def __save_metrics(self, df, limit, batch_size, name):
        df.to_csv(self.path(f'models/predicts/{limit}_{batch_size}_{name}.csv'), index=False)

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    # Upgrade to implementation from PyTorch
    def get_batch(self, batch_size: int) -> pd.DataFrame:
        batch = self.train[:batch_size]  # Получаем разметку и отправляем в размеченный набор данных
        self.train = self.train.drop(index=batch.index).reset_index(drop=True)
        return batch

    def start(self, batch_size: int, window: int = 3):
        if not self.classifier.history:  # Если модель пустая - добавляем данные
            self.classifier.add(self.init_df['phrase'].values, self.init_df['subtopic'].values)
            print('Холодный старт модели')

        people, model = 0, 0
        all_metrics, model_metrics, model_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                limit, all_predict = self.classifier.predict(self.train[TOPIC_IN_LB], self.limit)
                model += limit.shape[0]
                pred = pd.DataFrame({'item': self.train.iloc[limit]['item'],
                                     'topic': all_predict[limit] if limit.shape[0] else []})
                model_df = pd.concat([model_df, pred.explode('topic')], ignore_index=True)

                self.train = self.train.drop(index=limit).reset_index(drop=True)

            if len(self.lb.project.get_unlabeled_tasks_ids()) < 20:
                # Эмуляция разметки данных разметчиками
                batch = self.get_batch(batch_size=batch_size)
                people += batch.shape[0]
                self.lb.load_data(batch)
                self.train.to_csv('point/train.csv', index=False)  # POINT
                print('Данные загружены')

            self.lb.check_status()

            print('Получаем данные...')
            batch = self.lb.get_annotations()
            print('Данные получены...')

            # Оцениваем качество модели по батчам
            limit, all_predict = self.classifier.predict(batch[TEXT_IN_LB].values)
            metrics = self.classifier.metrics(batch[TOPIC_IN_LB].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = \
                limit.shape[0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            if all_metrics.shape[0] >= window:
                all_metrics.iloc[-1:, :3] = all_metrics.iloc[-window:, :3].agg('mean')

            if people >= 3000:
                self.run_model = True
                print('Запуск режима разметки моделью')
                print('Процесс калибровки порога...')

            self.classifier.add(batch[TEXT_IN_LB].values, batch[TOPIC_IN_LB].values)
            self.lb.save_point_tasks()  # POINT
            print(all_metrics.iloc[-1])

        self.__save_metrics(all_metrics, 0.97, batch_size, 'all_metrics')
        self.__save_metrics(model_metrics, 0.97, batch_size, 'model_metrics')
        self.__save_metrics(model_df, 0.97, batch_size, 'model_data')

    def controller(self, new: bool = False):
        """
        Решает проблему повторного запуска.
        """
        if new:
            pass  # Очистка POINT данных
        if 'train.csv' in os.listdir('point'):
            self.history = True
            self.train = self.__read_train('point/train.csv')
            self.classifier(history=True)
            self.lb()
        else:
            self.lb.create_project('Project 123')
            print('-' * 100)
            print(f'Создан новый проект')
        self.start(batch_size=500)


LABEL_STUDIO_URL = 'http://95.216.102.50:8083/'
API_KEY = '1145031409b0101a065785ccb7dedf49532e1172'
TOPIC_IN_LB = 'gender'
TEXT_IN_LB = 'text'
PATH_DF = 'run_data/data.csv'
PATH_DOMAIN = 'run_data/Domain.csv'

if __name__ == '__main__':
    system = ModelTraining(Classifier('models/adaptation/decorative_0_96_1_perfumery-adaptive.bin'),
                           LabelStudio(LABEL_STUDIO_URL, API_KEY))
    t1 = time()
    system.controller()
    print(time() - t1)
