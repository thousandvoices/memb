import _memb


class Builder:
    '''Builder object quantizes embeddings with given precision, creates index
    and saves it to file on request
    Parameters
    ----------
    dim : int
        Dimension of word vectors
    storage_type : str
        Type of storage for embeddings. Supported values are 'full', 'uniform' and
        'trained'
    bits_per_weight : int
        Number of bits used to represent single weight. If this value is beyond
        range accepted by quantization strategy, closest supported value will be
        used instead
    '''

    def __init__(self, dim, storage_type='trained', bits_per_weight=4):
        self._impl = _memb.Builder(dim, storage_type, bits_per_weight)

    def add_word(self, word, vector):
        '''Add word to builder
        Parameters
        ----------
        word : str
        
        vector : numpy.float32
        '''
        self._impl.add_word(word, vector)

    def save(self, filename):
        '''Compress builder content and save it to file
        Parameters
        ----------
        filename : str or pathlib.Path
        '''
        self._impl.save(str(filename))
