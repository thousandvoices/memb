from abc import ABC, abstractmethod
import _memb


class BaseReader(ABC):
    def __getitem__(self, key):
        '''Obtain vector representation for a word or a list of words
        Parameters
        ----------
        key: str of list of str
        '''
        if isinstance(key, str):
            return self.word_embedding(key)
        elif isinstance(key, list):
            return self.batch_embedding(key)
        else:
            raise TypeError('Key type is not supported')

    def to_keyed_vectors(self):
        '''Export model content to KeyedVectors object'''
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError('You must install gensim for KeyedVectors export')
        
        keyed_vectors = KeyedVectors(self.dim)
        words = self.keys()
        keyed_vectors.add(words, self.batch_embedding(words))

        return keyed_vectors

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def word_embedding(self, word):
        pass

    @abstractmethod
    def batch_embedding(self, words):
        pass

    @abstractmethod
    def tokenizer_embedding(self, tokenzer):
        pass


class Reader(BaseReader):
    '''Reader object allows to obtain embeddings for requested words quickly,
    reading and decoding them on the fly
    Parameters
    ----------
    filename : str or pathib.Path
    num_threads : int
        Number of threads used to decode large batches of words.
        Pass 0 to use as much threads as there are cores in the system
    Attributes
    ----------
    dim : int
        Embeddings dimension
    '''

    def __init__(self, filename, num_threads=4):
        super().__init__()
        self._impl = _memb.Reader(str(filename), num_threads)

    @property
    def dim(self):
        return self._impl.dim()
    
    def keys(self):
        '''List of words contained in model'''
        return self._impl.keys()

    def word_embedding(self, word):
        '''Obtain one-dimensional array of type float32 for a given word.
        If word is not present in the model, array filled with zeros is returned
        Parameters
        ----------
        word : str
        '''
        return self._impl.word_embedding(word)

    def batch_embedding(self, words):
        '''Obtain two-dimensional array of type float32 for a given list of words.
        Positions for words not present in the model are filled with zeros
        Parameters
        ----------
        words : list of str
        '''
        return self._impl.batch_embedding(words)

    def tokenizer_embedding(self, tokenizer):
        '''Convert keras.preprocessing.text.Tokenizer to weights of Embedding layer
        Parameters
        ----------
        tokenizer : keras.preprocessing.text.Tokenizer
        '''
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
