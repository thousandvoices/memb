import numpy as np
from .reader import BaseReader


class AverageUnionMaker:
    @staticmethod
    def check(readers):
        dims = [reader.dim for reader in readers]
        if any(dim != dims[0] for dim in dims):
            raise AssertionError('Dimensions of all readers must be equal for average mode')

    @staticmethod
    def dim(reader_dims):
        return reader_dims[0]

    @staticmethod
    def merge(vectors):
        return np.mean(vectors, axis=0)


class ConcatenatedUnionMaker:
    @staticmethod
    def check(readers):
        pass

    @staticmethod
    def dim(reader_dims):
        return sum(reader_dims)

    @staticmethod
    def merge(vectors):
        return np.concatenate(vectors, axis=-1)


UNION_MAKERS = {
    'average': AverageUnionMaker(),
    'concatenate': ConcatenatedUnionMaker(),
}


class ReadersUnion(BaseReader):
    def __init__(self, readers, mode):
        super().__init__()
        self._readers = readers
        self._union_maker = UNION_MAKERS.get(mode)
        
        if self._union_maker is None:
            raise KeyError('Mode {} is not supported. Available modes are {}'.format(
                mode, list(UNION_MAKERS.keys())))

        self._union_maker.check(readers)

    @property
    def dim(self):
        return self._union_maker.dim([reader.dim for reader in self._readers])

    def keys(self):
        all_keys = set()
        for reader in self._readers:
            all_keys |= set(reader.keys())
        
        return sorted(all_keys)

    def word_embedding(self, word):
        return self._union_maker.merge([reader.word_embedding(word) for reader in self._readers])

    def batch_embedding(self, words):
        return self._union_maker.merge([reader.batch_embedding(words) for reader in self._readers])

    def tokenizer_embedding(self, tokenizer):
        return self._union_maker.merge([reader.tokenizer_embedding(tokenizer) for reader in self._readers])
