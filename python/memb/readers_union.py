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
    '''ReadersUnion is a wrapper that makes a list of Readers behave just like
    one. It returns either average or concatenation of embeddings obtaied from
    the readers it contains.
    Parameters
    ----------
    readers : list of Reader

    mode : str
        Strategy to use for merging vectors. Can be either 'average' or 'concatenate'
    Attributes
    ----------
    dim : int
        Dimension of vectors after merge
    '''
    def __init__(self, readers, mode):
        super().__init__()

        if len(readers) < 2:
            raise AssertionError('You must pass at least 2 readers to create a union')

        self._union_maker = UNION_MAKERS.get(mode)
        if self._union_maker is None:
            raise KeyError('Mode {} is not supported. Available modes are {}'.format(
                mode, list(UNION_MAKERS.keys())))

        self._union_maker.check(readers)
        self._readers = readers

    @property
    def dim(self):
        return self._union_maker.dim([reader.dim for reader in self._readers])

    def keys(self):
        '''Union of keys contained in wrapped models'''
        all_keys = set()
        for reader in self._readers:
            all_keys |= set(reader.keys())
        
        return sorted(all_keys)

    def word_embedding(self, word):
        '''Merged vectors from all readers for a single word
        Parameters
        ----------
        word : str
        '''
        return self._union_maker.merge([reader.word_embedding(word) for reader in self._readers])

    def batch_embedding(self, words):
        '''Merged vectors from all readers for a list of words
        Parameters
        ----------
        words : list of str
        '''
        return self._union_maker.merge([reader.batch_embedding(words) for reader in self._readers])

    def tokenizer_embedding(self, tokenizer):
        '''Merged results of tokenizer_embedding call from all readers
        Parameters
        ----------
        tokenizer : keras.preprocessing.text.Tokenizer
        '''
        return self._union_maker.merge([reader.tokenizer_embedding(tokenizer) for reader in self._readers])
