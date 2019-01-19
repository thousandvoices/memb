from abc import ABC, abstractmethod
import _memb


class BaseReader(ABC):
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word_embedding(key)
        elif isinstance(key, list):
            return self.batch_embedding(key)
        else:
            raise TypeError('Key type is not supported')

    @abstractmethod
    def word_embedding(self, word):
        pass

    @abstractmethod
    def batch_embedding(self, words):
        pass


class Reader(BaseReader):
    def __init__(self, filename):
        super().__init__()
        self._impl = _memb.Reader(str(filename))

    @property
    def dim(self):
        return self._impl.dim()

    def word_embedding(self, word):
        return self._impl.word_embedding(word)

    def batch_embedding(self, words):
        return self._impl.batch_embedding(words)

    def tokenizer_embedding(self, tokenizer):
        word_indices = tokenizer.word_index.items()
        if tokenizer.num_words is not None:
            word_indices = [item for item in word_indices if item[1] < tokenizer.num_words]
            max_index = tokenizer.num_words
        else:
            max_index = max([item[1] for item in word_indices]) + 1

        sorted_word_list = [''] * max_index
        for word, idx in word_indices:
            sorted_word_list[idx] = word

        return self.batch_embedding(sorted_word_list)
