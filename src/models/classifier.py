import os
import pickle
import faiss
import fasttext
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.metrics import precision_score, recall_score, f1_score


# Add a logger
class PredictError(Exception):
    def __init__(self, text):
        self.txt = text


class Classifier:
    history = False
    y = np.array([])
    emb = {}

    def __init__(self, model: str, faiss_path: str = None, embedding_path: str = None):
        """
        :param model: Путь до модели построения эмбэдингов.
        :param faiss_path: Путь до сохранённых индексов faiss.
        :param embedding_path: Путь до сохранённых вектороных представлений фраз.
        """
        print('Загрузка модели...')
        self.model = fasttext.load_model(str(self.path(model)))
        print('Модель успешно загружена')
        # if faiss_path:
        #     with open(self.path(faiss_path), 'rb') as f:
        #         self.index, self.y = pickle.load(f)
        #         self.history = 1
        # if embedding_path:
        #     with open(self.path(embedding_path), 'rb') as f:
        #         self.emb = pickle.load(f)

    def __call__(self, history: bool, *args, **kwargs):
        self.history = history
        if self.history:
            with open(self.path('point/faiss.pkl'), 'rb') as f:
                self.index, self.y = pickle.load(f)
            with open(self.path('point/emb.pkl'), 'rb') as f:
                self.emb = pickle.load(f)
            print('Загружен последний статус модели')

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def embeddings(self, texts: list or np.array) -> np.array:
        """
        Построение векторынх представлений для фраз.
        :param texts: Набор фраз.
        :return: Набор векторов.
        """
        emb = []
        for text in texts:
            text = text.replace('-', ' ').lower().strip()
            if not self.emb.get(text, np.array([])).shape[0]:
                self.emb[text] = normalize([self.model.get_sentence_vector(text)])[0]
            emb.append(self.emb.get(text))
        with open(self.path('point/emb.pkl'), 'wb') as f:
            pickle.dump(self.emb, f)
        return np.array(emb, dtype='float32')

    def add(self, x: np.array, y: np.array):
        """
        Добавление фраз в индекс и сохранение текущего состояния модели.
        :param x: Набор фраз.
        :param y: Набор категорий.
        :return: self.
        """
        # Удаление неразмеченных данных, так как пустые значения не нужны в индексе.
        nan_index = [i for i, c in enumerate(y) if type(c) == float]
        x = np.delete(x, nan_index, axis=0)
        y = np.delete(y, nan_index, axis=0)

        if not self.y.shape[0]:
            self.index = faiss.IndexFlat(self.embeddings(x[0]).shape[1])

        self.index.add(self.embeddings(x))
        self.y = np.append(self.y, y).reshape(-1, 1)
        with open(self.path('point/faiss.pkl'), 'wb') as f:
            pickle.dump((self.index, self.y), f)

    @staticmethod
    def __get_top_classes(classes: np.array, max_count: int = 10) -> list:
        '''
        Возвращает n-e количество предсказанных классов моделью.
        :param classes: Классы из которых небходимо выбрать топ n.
        :param max_count: Максимальное количество возвращаемых классов.
        :return: Список классов для текущего объекта.
        '''
        unique = set()
        for subtopic in classes:
            unique = unique.union(set(subtopic))
            if len(unique) >= max_count:
                break
        return list(unique)[:max_count]

    def predict(self, x: np.array, limit: float = 1) -> tuple:
        """
        Предсказание категории для фразы.
        :param x: Набор фраз.
        :param limit: Допустимая дистанция.
        :return: Словарь из двух элементов (predict_limit, all_predict),
        где predict_limit - это индексы объектов, которые прдесказаны по дистанции,
        all_predict - масиив, который для фразы содержит список катеогрий.
        """
        predict_limit, all_predict = [], []
        dis, ind = self.index.search(self.embeddings(x), k=25)
        for i in range(x.shape[0]):
            if any(dis[i] <= 1 - limit):  # We save indexes where the models is not sure
                predict_limit.append(i)
                # Consider the weighted confidence of classes
                all_predict.append(self.__get_top_classes(self.y[ind[i][dis[i] <= 1 - limit]]))
            else:
                all_predict.append(self.__get_top_classes(self.y[ind[i]]))
        return np.array(predict_limit), np.array(all_predict, dtype='object')

    @staticmethod
    def metrics(y_true: np.array, y_pred: np.array, average: str = 'samples') -> pd.DataFrame:
        """
        Метод выполняет подсчёт метрик.
        :param y_true: Истинное значение целевой переменной.
        :param y_pred: Предсказанное значение целевой переменной.
        :param average: Метод подсчёта метрик.
        :return: Результаты метрик в формате pd.DataFrame.
        """
        # Удаление неразмеченных данных, так как по ним невозможно снять метрики.
        nan_index = [i for i, c in enumerate(y_true) if type(c) == float]
        y_true = np.delete(y_true, nan_index, axis=0)
        y_pred = np.delete(y_pred, nan_index, axis=0)

        classes = set()  # Отбор уникальных классов
        for i in range(y_true.shape[0]):
            classes = classes | set(y_true[i]) | set(y_pred[i])
        average = 'weighted' if len(classes) == 1 else average

        mlb = MultiLabelBinarizer(classes=list(classes))
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)
        return pd.DataFrame({
            'f1': [f1_score(y_true, y_pred, average=average)],
            'precision': [precision_score(y_true, y_pred, average=average, zero_division=1)],
            'recall': [recall_score(y_true, y_pred, average=average, zero_division=0)],
            'validation_size': [y_true.shape[0]]
        })
