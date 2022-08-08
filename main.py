import os
import pandas as pd
import numpy as np
from time import time
from pathlib import Path
from src.data import CreateModelData

# ____________
import torch
import warnings
import torch.nn.functional as F
import transformers

from transformers import BertModel, BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textwrap import wrap
from tqdm.notebook import tqdm
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Torch settings
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 777
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# BERT settings
MAX_LEN = 254
BATCH_SIZE = 32
PRE_TRAINED_MODEL_NAME = 'cointegrated/rubert-tiny'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)


class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)}


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df.phrase.to_numpy(),
        targets=df.true.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(ds, batch_size=batch_size)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outputs["pooler_output"])
        return self.out(output)


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def run_train(train_data_loader, val_data_loader, df_train, df_val, epochs=5):
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in tqdm(range(epochs)):
        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device,
                                            len(df_train))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc


class ModelTraining:
    run_model = False

    def __init__(self, classifier: SentimentClassifier):
        self.classifier = classifier
        self.train = self.__read_train('data/processed/marked-up-join.csv')
        self.init_df = pd.read_csv('data/processed/init_df.csv')
        self.init_size = self.init_df.shape[0]

        self.full = pd.read_csv('data/raw/Decorative/Full_test.csv')

    def __read_train(self, train_file: str):
        """
        Загрузка набора данных для снятия метрик.
        :param train_file: Путь до файла.
        :return: Агрегированный набор данных.
        """
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
        self.init_df = pd.concat([self.init_df, markup], ignore_index=True)
        self.init_df.to_csv(self.path('data/processed/init_df.csv'))

    @staticmethod
    def __drop_full_from_train(train, df_drop):
        train.reset_index(inplace=True)
        drop = df_drop.merge(train, left_on='phrase', right_on='phrase')['index'].values
        return train.set_index('index').drop(index=drop)

    def __save_metrics(self, df, limit, batch_size, name):
        df.to_csv(self.path(f'models/predicts/{limit}_{batch_size}_{name}.csv'), index=False)

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
        group_all_df = self.init_df.groupby(by='phrase').agg(
            subtopic=('subtopic', 'unique'),
            true=('true', 'unique')).reset_index()

        # Стартовое обучение модели
        train_data_loader = create_data_loader(group_all_df, tokenizer, MAX_LEN, BATCH_SIZE)
        run_train(train_data_loader, train_data_loader, group_all_df, group_all_df)


        people, model = 0, 0
        all_metrics, model_metrics, model_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        while self.train.shape[0]:
            if self.run_model:
                # Размечаем набор данных моделью
                index_limit, all_predict = self.classifier.predict(self.full['item'], limit)
                model += index_limit.shape[0]
                pred = pd.DataFrame({'phrase': self.full.iloc[index_limit].item,
                                     'subtopic': all_predict[index_limit] if
                                     index_limit.shape[0] else []})
                model_df = pd.concat([model_df, pred.explode('subtopic')], ignore_index=True)

                # Оцениваем качество модели, если количество предсказанных объектов больше 10
                # if index_limit.shape[0] > 10:
                #     metrics = self.clas.metrics(pred['true'].values, pred['subtopic'].values)
                #     metrics[['model_from_val', 'model_from_all', 'people_from_val']] = \
                #         index_limit.shape[0], model, people
                #     model_metrics = pd.concat([model_metrics, metrics])
                #     model_metrics.iloc[-1:, :3] = model_metrics.iloc[-window:, :3].agg('mean')
                self.train = self.__drop_full_from_train(self.train, pred)
                self.full = self.full.drop(index=index_limit).reset_index(drop=True)
                # self.__update_predict_df(pred.explode('subtopic').explode('true'))

            # Эмуляция разметки данных разметчиками
            batch = self.batch(batch_size=batch_size)
            people += batch.shape[0]

            # Оцениваем качество модели по батчам
            index_limit, all_predict = self.classifier.predict(batch['phrase'].values, limit)
            metrics = self.classifier.metrics(batch['true'].values, all_predict)
            metrics[['model_from_val', 'model_from_all', 'people_from_val']] = index_limit.shape[
                                                                                   0], model, people
            all_metrics = pd.concat([all_metrics, metrics])
            all_metrics.iloc[-1:, :3] = all_metrics.iloc[-window:, :3].agg('mean')
            if people >= 3000:
                self.run_model = True

            # Добавляем новые индексы в модель
            group_all_df = self.init_df[self.init_size:].groupby(by='phrase').agg(
                subtopic=('subtopic', 'unique'),
                true=('true', 'unique')).reset_index()
            self.classifier.add(group_all_df['phrase'], group_all_df['subtopic'])
            self.init_size = self.init_df.shape[0]  # Обновляем размер набора данных
            print(all_metrics.iloc[-1])

        self.__save_metrics(all_metrics, limit, batch_size, 'all_metrics')
        self.__save_metrics(model_metrics, limit, batch_size, 'model_metrics')
        self.__save_metrics(model_df, limit, batch_size, 'model_data')


if __name__ == '__main__':
    preproc = CreateModelData('data/raw/Decorative/Domain.csv')
    preproc.join_train_data('data/raw/Decorative/Synonyms_test.csv',
                            'data/raw/Decorative/Full_test.csv')

    model = SentimentClassifier(len(preproc.classes))
    model = model.to(device)

    EPOCHS = 2

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)

    print('Формирование данных завершено')
    system = ModelTraining(model)
    t1 = time()
    system.start(limit=0.97, batch_size=500)
    print(time() - t1)
