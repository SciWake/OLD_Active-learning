import os
import pickle
import pandas as pd
import numpy as np
from label_studio_sdk import Client
from pathlib import Path
from time import sleep


class LabelStudio:
    tasks_ids, text_in_lb = set(), set()

    def __init__(self, label_studio_url: str, api_key: str):
        self.__client = Client(url=label_studio_url, api_key=api_key)
        try:
            self.project = self.client.get_projects()[0]
        except Exception:
            print('Нет проектов, ошибка загрузки...')

    def __call__(self, *args, **kwargs):
        with open(self.path('point/LabelStudio.pkl'), 'rb') as f:
            self.tasks_ids, self.text_in_lb = pickle.load(f)

    @property
    def client(self) -> Client:
        return self.__client

    @staticmethod
    def path(path):
        return Path(os.getcwd(), path)

    def create_project(self, name: str):
        with open(self.path('src/labelstud/labeling_config.html'), 'r', encoding='UTF-8') as f:
            config = f.read()
        with open(self.path('src/labelstud/instruction.html'), 'r', encoding='UTF-8') as f:
            instruction = f.read()
        self.client.start_project(title=name, label_config=config, expert_instruction=instruction)
        self.project = self.client.get_projects()[0]

    def search_project(self, project_name: str):
        for project in self.client.get_projects():
            if project.title == project_name:
                return project

    def load_data(self, data: pd.DataFrame, column: str = 'item'):
        load = []
        for task in data[column]:
            if task not in self.text_in_lb:
                load.append({'text': task})
                self.text_in_lb.add(task)
        self.project.import_tasks(load)
        self.save_point_tasks()

    def check_status(self):
        while len(self.project.get_unlabeled_tasks_ids()) >= 495:
            sleep(10)
            print('Ожидание разметки...')

    def save_point_tasks(self):
        """
        Сохранение данных, которые были получены из API.
        """
        with open(self.path('point/LabelStudio.pkl'), 'wb') as f:  # POINT
            pickle.dump((self.tasks_ids, self.text_in_lb), f)

    def get_annotations(self):
        annotations = pd.DataFrame()
        for task in self.project.export_tasks():
            if int(task['id']) in self.tasks_ids:
                continue

            a = {i['from_name']: [i['value']['choices']] for i in task['annotations'][0]['result']}
            df = pd.DataFrame({'id': [task['id']],
                               'text': [task['data']['text']],
                               **a})
            annotations = pd.concat([annotations, df], ignore_index=True)
            self.tasks_ids.add(int(task['id']))
        return annotations
