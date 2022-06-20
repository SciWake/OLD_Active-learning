import pandas as pd
import pymorphy2
from functools import lru_cache
from multiprocessing import Pool
import tqdm
import re

# pymorphy2 - библиотека методов для морфологического анализа (в том числе лемматизации) русскоязычного текста
m = pymorphy2.MorphAnalyzer()

# убираем все небуквенные символы
regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")


def words_only(text, regex=regex):
    try:
        return regex.findall(text)
    except:
        return []


# если вы работаете не колабе, можно заменить pymorphy на mystem и раскомментирвать первую строку про lru_cache
@lru_cache(maxsize=128)
def lemmatize(text, pymorphy=m):
    try:
        return " ".join([pymorphy.parse(w)[0].normal_form for w in text])
    except:
        return " "


def clean_text(text):
    return lemmatize(words_only(text))


# загрузим и посмотрим на наш датасет

# загружаем положительные твитты
positive = pd.read_csv('positive.csv', sep=';', usecols=[3], names=['text'])
positive['label'] = ['positive'] * len(positive)  # расставляем метки

# загружаем отрицательные твитты
negative = pd.read_csv('negative.csv', sep=';', usecols=[3], names=['text'])
negative['label'] = ['negative'] * len(negative)  # расставляем метки

# соединяем два набора данных
data = positive.append(negative)
data.head()

if __name__ == '__main__':
    with Pool(8) as p:
        lemmas = list(
            tqdm.tqdm(p.imap(clean_text, data['text']), total=len(data)))
    data['lemmas'] = lemmas
    data.head()

    compression_opts = dict(method='zip', archive_name='out.csv')
    data.to_csv('out.zip', index=False, compression=compression_opts)
