import _memb
import numpy as np


class Builder:
    def __init__(self, dim, storage_type='full'):
        self._impl = _memb.Builder(dim, storage_type)

    def add_word(self, word, vector):
        self._impl.add_word(word, vector)

    def save(self, filename):
        self._impl.save(str(filename))
        

class Reader:
    def __init__(self, filename):
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


class ConcatenatingReader:
    def __init__(self, readers):
        self._readers = readers

    @property
    def dim(self):
        return sum([reader.dim for reader in self._readers])

    def word_embedding(self, word):
        return np.concatenate([reader.word_embedding(word) for reader in self._readers], axis=-1)

    def batch_embedding(self, words):
        return np.concatenate([reader.batch_embedding(words) for reader in self._readers], axis=-1)

    def tokenizer_embedding(self, tokenizer):
        return np.concatenate([reader.tokenizer_embedding(tokenizer) for reader in self._readers], axis=-1)
