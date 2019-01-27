import _memb


class Builder:
    def __init__(self, dim, storage_type='trained', bits_per_weight=4):
        self._impl = _memb.Builder(dim, storage_type, bits_per_weight)

    def add_word(self, word, vector):
        self._impl.add_word(word, vector)

    def save(self, filename):
        self._impl.save(str(filename))
