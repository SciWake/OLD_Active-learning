import re
import numpy as np
import pymorphy2
from functools import lru_cache


class LemmaPredictText:
    pymorphy = pymorphy2.MorphAnalyzer()

    def __init__(self, regex: str = "[А-ЯЁа-яё]+"):
        self.regex = re.compile(regex)

    def words_only(self, text: str) -> list:
        try:
            return self.regex.findall(text.lower())
        except AttributeError:
            return []

    @lru_cache(maxsize=128)
    def lemma(self, text: list) -> str:
        try:
            return " ".join(
                [self.pymorphy.parse(w)[0].normal_form for w in text])
        except AttributeError:
            return " "

    def clean_text(self, text):
        return self.lemma(self.words_only(text))


class ClearingPhrases(LemmaPredictText):
    def __init__(self, texts: np.array, regex: str = "[А-ЯЁа-яё]+"):
        super(ClearingPhrases, self).__init__(regex)
        self.texts = texts

    @property
    def get_best_texts(self) -> list:
        texts = []
        for text in self.texts:
            split_w = self.words_only(text)
            exit = 0
            if len(split_w) >= 2:
                for word in split_w:
                    if len(word) <= 6:
                        exit = 1
                        break
                if exit:
                    continue
                texts.append(' '.join(split_w))
        return list(set(texts))
