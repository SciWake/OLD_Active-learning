import pandas as pd
from label_studio_sdk import Client
from label_studio_sdk import project


class LabelStudio:
    project = None

    def __init__(self, label_studio_url: str, api_key: str):
        self.__client = Client(url=label_studio_url, api_key=api_key)

    @property
    def client(self) -> Client:
        return self.__client

    def create_project(self, name: str):
        with open('labeling_config.html', 'r', encoding='UTF-8') as f:
            config = f.read()
        with open('instruction.html', 'r', encoding='UTF-8') as f:
            instruction = f.read()
        self.client.start_project(title=name, label_config=config, expert_instruction=instruction)
        self.project = self.client.get_projects()[0]

    def search_project(self, project_name: str):
        for project in self.client.get_projects():
            if project.title == project_name:
                return project

    def load_data(self, project: project, file: str, column: str):
        # project.import_tasks([task['data'] for task in tasks[50:60]])
        pass

    def get_project_info(self):
        pass


if __name__ == '__main__':
    LABEL_STUDIO_URL = ''
    API_KEY = ''
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
