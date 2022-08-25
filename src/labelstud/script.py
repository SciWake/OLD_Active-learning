import os
import pandas as pd
from label_studio_sdk import Client
from label_studio_sdk import project
from pathlib import Path


class LabelStudio:
    project = None

    def __init__(self, label_studio_url: str, api_key: str):
        self.__client = Client(url=label_studio_url, api_key=api_key)

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
        self.project.import_tasks([{'text': task} for task in data[column]])

    def get_project_info(self):
        pass


if __name__ == '__main__':
    LABEL_STUDIO_URL = ''
    API_KEY = ''
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
