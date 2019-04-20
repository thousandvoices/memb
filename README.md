
## Motivation
Even though vector representations of words are an important concept in natural language processing and maintain
high popularity since introduction of word2vec, existing formats for storing them still suffer from several drawbacks.
This library addresses two of them: large file size and slow loading into memory.

To reduce storage space requirements, we use aggressive quantization followed by Huffman encoding. The implementation
is mostly based on [Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization and Huffman
Coding](https://arxiv.org/pdf/1510.00149.pdf) paper. Experiments show that quality loss remains negligible even for 
models compressed by a factor of 6.

To overcome second problem, we use a single `mmap` call during `Reader` construction for disk input. It allows to
* Reduce initial load time to several milliseconds
* Make all data loads lazy. We only read vectors requested by user
* Keep data in physical memory even after process restart. On the other hand, when the machine is low on free memory,
data will be evicted by operating system even without explicit deletion and garbage collection calls.

## Experiments
We present results for tayga_upos_skipgram_300_2_2019 model in this section. We also observed similar behavior for 
ruscorpora_upos_cbow_300_20_2019 model. You can see that
* A compressed model using 6 bits per weight is 6 times smaller than full-precision one and is indistinguishable 
from it in terms of quality
* The correlation between hand-labeled and predicted word similarity is much less sensitive to quantization noise 
that analogy task accuracy. Even models that use 2 bits per weight show correlation just as high as full models
* Models compressed to 1 bit per weight still maintain significant predictive power

<p float="left">
  <img src="https://github.com/thousandvoices/memb/raw/master/docs/images/spearman.png" alt="spearman" width="400" />
  <img src="https://github.com/thousandvoices/memb/raw/master/docs/images/analogy.png" alt="analogy" width="400" />
  <img src="https://github.com/thousandvoices/memb/raw/master/docs/images/sizes.png" alt="size" width="400" />
</p>

Experiments involving training neural networks with compressed embeddings show that 4 bits of precision is sufficient
to match full models.

## Quickstart
* Download and install wheels from [releases page](https://github.com/thousandvoices/memb/releases)
* Obtain compressed embedding files:
  * Download pretrained vectors from https://drive.google.com/open?id=1g2Bv-5qscDRdnD4UjE6TAB2uaNdz2JdU.
  Glove 840B, fasttext crawl (english), tayga_upos_skipgram_300_2_2019 and ruscorpora_upos_cbow_300_20_2019 
  (russian from [RusVectores](https://rusvectores.org/ru/models/) project) are available at the moment.
  * Or use memb_converter tool to convert embeddings from word2vec text format. It is recommended to pass 
  `--quantization trained --bits-per-weight 6` to the script as parameters and leave `--max-words` empty.
* Now you can create a `Reader` object:
```python
from memb import Reader

reader = Reader('glove.840B.300d.4bit.bin')
```
  * Find out vectors dimensions:
```python
print(reader.dim)
```
  * Obtain embedding for a single word, which returns an array of shape (reader.dim,):
```python
print(reader['the'])
```
  * Or embeddings for a list of words, which is a two-dimensional array of shape (n_words, reader.dim):
```python
print(reader[['a', 'the', 'of']])
```

## Additional features
[`ReadersUnion`](https://github.com/thousandvoices/memb/blob/master/python/memb/readers_union.py#L41) allows to combine 
several embedding sources into a single one. Use `mode='average'` or `mode='concatenate'` to choose respective merging 
strategy.
```python
from memb import Reader, ReadersUnion

filenames = ['glove.840B.300d.4bit.bin', 'fasttext-crawl-300d-2M.4bit.bin']
readers = [Reader(filename) for filename in filenames]

union = ReadersUnion(readers, mode='concatenate')
print(union['the'])
```

[`tokenizer_embedding`](https://github.com/thousandvoices/memb/blob/master/python/memb/reader.py#L94) method creates 
an array suitable for using as `Embedding` layer weights from 
[`keras.preprocessing.text.Tokenizer`](https://keras.io/preprocessing/text/)
```python
from memb import Reader
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding

texts = ['First sentence', 'Second sentence']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

reader = Reader('glove.840B.300d.4bit.bin')

embedding_matrix = reader.tokenizer_embedding(tokenizer)
embedding_layer = Embedding(
    *embedding_matrix.shape,
    weights=[embedding_matrix],
    trainable=False)
```

Use `to_keyed_vectors` method to export model to [`KeyedVectors`](https://radimrehurek.com/gensim/models/keyedvectors.html).
Note: you must install gensim to use it.

## Building from source
To build library from source, you'll need
* CMake
* Compiler with C++14 support
* Additional libraries
  * Boost
  * Flatbuffers

You can follow steps described in dependency installation scripts for [linux](https://github.com/thousandvoices/memb/blob/master/tools/development_image/install_deps_linux.sh) and [mac os](https://github.com/thousandvoices/memb/blob/master/tools/development_image/install_deps_darwin.sh).
